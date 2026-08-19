from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class TokenBucket:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.capacity = float(requests_per_minute)
        self.refill_per_second = requests_per_minute / 60
        self.tokens = self.capacity
        self.updated_at = clock()
        self.clock = clock
        self.sleeper = sleeper
        self.lock = asyncio.Lock()

    async def acquire(self, *, exempt: bool = False) -> None:
        if exempt:
            return
        while True:
            async with self.lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.refill_per_second)
                self.updated_at = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.refill_per_second
            await self.sleeper(wait)
