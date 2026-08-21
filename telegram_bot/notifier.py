from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import httpx


def _parse_chat_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError:
        return ()


class PipelineNotifier:
    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, *, min_interval: float = 0.05, update_interval: float = 60.0) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_ids = _parse_chat_ids(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""))
        self._client = httpx.Client(timeout=15.0) if token and self._chat_ids else None
        self._url = f"{self.BASE_URL.format(token=token)}/{{}}" if self._client else ""
        self._min_interval, self._update_interval = max(0.0, min_interval), max(5.0, update_interval)
        self._last_send = 0.0
        self._send_lock, self._state_lock = threading.Lock(), threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status, self._started = "", 0.0
        self._message_ids: dict[int, int] = {}

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _post(self, method: str, *, json: dict | None = None, data: dict | None = None, files: dict | None = None):
        if not self._client:
            return None
        with self._send_lock:
            delay = self._min_interval - (time.monotonic() - self._last_send)
            if delay > 0:
                time.sleep(delay)
            try:
                response = self._client.post(self._url.format(method), json=json, data=data, files=files)
                self._last_send = time.monotonic()
                response.raise_for_status()
                return response.json().get("result")
            except (httpx.HTTPError, ValueError):
                self._last_send = time.monotonic()
                return None

    def notify(self, text: str) -> None:
        for chat_id in self._chat_ids:
            self._post("sendMessage", json={"chat_id": chat_id, "text": text[:4000]})

    def _publish_status(self) -> None:
        with self._state_lock:
            text = f"{self._status}\nElapsed: {int(time.monotonic() - self._started)}s"[:4000]
        for chat_id in self._chat_ids:
            message_id = self._message_ids.get(chat_id)
            result = (
                self._post("editMessageText", json={"chat_id": chat_id, "message_id": message_id, "text": text})
                if message_id
                else None
            )
            if result is None:
                result = self._post("sendMessage", json={"chat_id": chat_id, "text": text})
            if result:
                self._message_ids[chat_id] = int(result["message_id"])

    def start_updates(self, status: str) -> None:
        if not self.enabled:
            return
        self.stop_updates()
        with self._state_lock:
            self._status, self._started = status, time.monotonic()
        self._message_ids.clear()
        self._stop.clear()
        self._publish_status()
        self._thread = threading.Thread(target=self._update_loop, name="telegram-pipeline-updates", daemon=True)
        self._thread.start()

    def update(self, status: str) -> None:
        with self._state_lock:
            self._status = status

    def _update_loop(self) -> None:
        while not self._stop.wait(self._update_interval):
            self._publish_status()

    def stop_updates(self, final_status: str | None = None) -> None:
        thread, self._thread = self._thread, None
        self._stop.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if final_status and self.enabled:
            self.update(final_status)
            self._publish_status()

    def notify_phase(self, phase: str, symbol: str, detail: str = "") -> None:
        text = f"{phase} | {symbol}" + (f"\n{detail}" if detail else "")
        self.update(text)
        self.notify(text)

    def _send_file(self, method: str, field: str, path: str | Path, caption: str) -> None:
        source = Path(path)
        if not self.enabled or not source.is_file():
            return
        for chat_id in self._chat_ids:
            try:
                with source.open("rb") as stream:
                    self._post(
                        method,
                        data={"chat_id": chat_id, "caption": caption[:1024]},
                        files={field: (source.name, stream)},
                    )
            except OSError:
                continue

    def send_photo(self, path: str | Path, caption: str = "") -> None:
        self._send_file("sendPhoto", "photo", path, caption)

    def send_document(self, path: str | Path, caption: str = "") -> None:
        self._send_file("sendDocument", "document", path, caption)

    def close(self) -> None:
        self.stop_updates()
        if self._client:
            self._client.close()
            self._client = None
