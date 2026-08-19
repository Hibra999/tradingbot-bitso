from __future__ import annotations

import asyncio
import inspect
import json
import random
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from .bitso_rest import BitsoRESTClient
from .order_book import LocalOrderBook, SequenceGapError


class BitsoWebSocketClient:
    def __init__(
        self,
        rest: BitsoRESTClient,
        books: tuple[str, ...],
        websocket_url: str,
        *,
        on_book_update: Callable[[LocalOrderBook], Any] | None = None,
    ):
        self.rest = rest
        self.websocket_url = websocket_url
        self.books = {book: LocalOrderBook(book) for book in books}
        self.on_book_update = on_book_update
        self._stopping = asyncio.Event()

    async def _emit(self, book: LocalOrderBook) -> None:
        if self.on_book_update:
            result = self.on_book_update(book)
            if inspect.isawaitable(result):
                await result

    @staticmethod
    async def _receive(socket, queue: asyncio.Queue[dict]) -> None:
        async for raw in socket:
            message = json.loads(raw)
            if message.get("type") != "ka":
                await queue.put(message)

    async def _resync(self, book: str) -> None:
        snapshot = await self.rest.request("GET", "/order_book", params={"book": book, "aggregate": "false"})
        self.books[book].bootstrap(snapshot)
        await self._emit(self.books[book])

    async def _connection(self) -> None:
        async with websockets.connect(self.websocket_url, ping_interval=20, ping_timeout=20) as socket:
            queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10_000)
            receiver = asyncio.create_task(self._receive(socket, queue))
            try:
                for book in self.books:
                    await socket.send(json.dumps({"action": "subscribe", "book": book, "type": "diff-orders"}, separators=(",", ":")))
                await asyncio.gather(*(self._resync(book) for book in self.books))
                while not self._stopping.is_set():
                    message = await queue.get()
                    if message.get("type") != "diff-orders" or message.get("book") not in self.books:
                        continue
                    book = self.books[message["book"]]
                    try:
                        changed = book.apply_diff(message)
                    except SequenceGapError:
                        await self._resync(book.book)
                        continue
                    if changed:
                        await self._emit(book)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    async def run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                await self._connection()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                delay = min(30.0, 2**attempt) + random.random()
                attempt = min(attempt + 1, 5)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=delay)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stopping.set()
