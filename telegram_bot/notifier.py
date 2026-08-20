"""Lightweight synchronous Telegram notifier for pipeline progress.

This module is designed to work *outside* the async live-service context.
It uses ``httpx`` (already a project dependency) to send plain messages and
photos to the Telegram Bot API without pulling in the full
``python-telegram-bot`` stack or requiring an event loop.

If ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_ALLOWED_CHAT_IDS`` are not set, every
method silently no-ops so the pipeline keeps running without Telegram.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

import httpx


def _parse_chat_ids(raw: str) -> list[int]:
    """Parse a comma-separated string of chat IDs into a list of ints."""
    ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids


class PipelineNotifier:
    """Send pipeline progress updates to Telegram chats.

    Instantiation never raises -- if credentials are missing the notifier
    becomes a silent no-op.
    """

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, *, min_interval: float = 1.0) -> None:
        self._token: Optional[str] = None
        self._chat_ids: list[int] = []
        self._enabled = False
        self._min_interval = min_interval
        self._last_send: float = 0.0
        self._client: Optional[httpx.Client] = None

        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_ids_raw = (os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip()

        if not token or not chat_ids_raw:
            return

        try:
            self._chat_ids = _parse_chat_ids(chat_ids_raw)
        except ValueError:
            return

        if not self._chat_ids:
            return

        self._token = token
        self._enabled = True
        self._client = httpx.Client(timeout=15.0)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _throttle(self) -> None:
        """Ensure we don't exceed Telegram's rate limits."""
        elapsed = time.monotonic() - self._last_send
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_send = time.monotonic()

    def _url(self, method: str) -> str:
        return f"{self.BASE_URL.format(token=self._token)}/{method}"

    def notify(self, text: str) -> None:
        """Send a text message to all allowed chats."""
        if not self._enabled or not self._client:
            return
        self._throttle()
        for chat_id in self._chat_ids:
            try:
                self._client.post(
                    self._url("sendMessage"),
                    json={"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"},
                )
            except Exception:
                continue

    def send_photo(self, path: Union[str, Path], caption: str = "") -> None:
        """Send a photo file to all allowed chats."""
        if not self._enabled or not self._client:
            return
        path = Path(path)
        if not path.exists():
            return
        self._throttle()
        for chat_id in self._chat_ids:
            try:
                with open(path, "rb") as f:
                    self._client.post(
                        self._url("sendPhoto"),
                        data={"chat_id": chat_id, "caption": caption[:1024]},
                        files={"photo": (path.name, f, "image/png")},
                    )
            except Exception:
                continue

    def notify_phase(self, phase: str, symbol: str, detail: str = "") -> None:
        """Send a formatted phase-progress message."""
        text = f"<b>{phase}</b> | {symbol}"
        if detail:
            text += f"\n{detail}"
        self.notify(text)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
