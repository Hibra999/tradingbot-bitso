from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


class BacktestManager:
    """One bounded research subprocess; arguments never pass through a shell."""

    def __init__(self, repository: str | Path, *, timeout_seconds: float = 21_600):
        self.repository = Path(repository).resolve()
        self.timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()

    async def run(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._lock.locked():
            return {"status": "busy"}
        profile = request.get("profile", "smoke")
        symbol = request.get("symbol")
        if profile not in {"smoke", "full"} or symbol not in {None, "BTC/USD", "ETH/USD"}:
            raise ValueError("unsupported backtest parameters")
        command = [sys.executable, "run_quant_pipeline.py", "--profile", profile]
        if symbol:
            command.extend(("--symbol", symbol))
        async with self._lock:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repository,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                return_code = await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                return {"status": "timeout"}
        return {"status": "complete" if return_code == 0 else "failed", "return_code": return_code}
