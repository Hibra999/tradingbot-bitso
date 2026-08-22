from __future__ import annotations

import html
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


def _redacted_error(error: BaseException) -> str:
    detail = f"{type(error).__name__}: {error}"
    secret_markers = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")
    for name, value in os.environ.items():
        if value and any(marker in name.upper() for marker in secret_markers):
            detail = detail.replace(value, "[redacted]")
    return detail[:3000]


class PipelineNotifier:
    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, *, min_interval: float = 0.05, update_interval: float = 10.0) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_ids = _parse_chat_ids(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""))
        self._client = httpx.Client(timeout=15.0) if token and self._chat_ids else None
        self._url = f"{self.BASE_URL.format(token=token)}/{{}}" if self._client else ""
        self._min_interval, self._update_interval = max(0.0, min_interval), max(5.0, update_interval)
        self._last_send = 0.0
        self._send_lock, self._state_lock = threading.Lock(), threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status, self._query_status, self._started = "", "", 0.0
        self._message_ids: dict[int, int] = {}
        self._sent_message_ids: dict[int, set[int]] = {}
        self._command_offset: int | None = None

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

    def _remember_message(self, chat_id: int, result: object) -> None:
        if not isinstance(result, dict) or "message_id" not in result:
            return
        try:
            message_id = int(result["message_id"])
        except (TypeError, ValueError):
            return
        with self._state_lock:
            self._sent_message_ids.setdefault(chat_id, set()).add(message_id)

    def _send_message(self, chat_id: int, text: str, *, parse_mode: str | None = None) -> None:
        payload: dict[str, object] = {"chat_id": chat_id, "text": text[:4000]}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        self._remember_message(chat_id, self._post("sendMessage", json=payload))

    def notify(self, text: str) -> None:
        for chat_id in self._chat_ids:
            self._send_message(chat_id, text)

    def notify_html(self, text: str) -> None:
        for chat_id in self._chat_ids:
            self._send_message(chat_id, text, parse_mode="HTML")

    def _status_text(self, *, query: bool = False) -> str:
        with self._state_lock:
            elapsed = int(time.monotonic() - self._started)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            status = self._query_status if query else self._status
        return f"QUANT PIPELINE\n{status}\n\nElapsed  {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _publish_status(self, chat_ids: tuple[int, ...] | None = None) -> None:
        status = self._status_text().removeprefix("QUANT PIPELINE\n")
        text = f"<b>QUANT PIPELINE</b>\n<pre>{html.escape(status)}</pre>"[:4000]
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "Progress", "callback_data": "pipeline:progress"},
                    {"text": "Status", "callback_data": "pipeline:status"},
                ],
                [
                    {"text": "Help", "callback_data": "pipeline:help"},
                    {"text": "Clear", "callback_data": "pipeline:clear"},
                ],
            ]
        }
        for chat_id in chat_ids or self._chat_ids:
            with self._state_lock:
                message_id = self._message_ids.get(chat_id)
            payload: dict[str, object] = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "reply_markup": reply_markup,
            }
            if message_id:
                payload["message_id"] = message_id
            result = (
                self._post("editMessageText", json=payload)
                if message_id
                else None
            )
            if result is None:
                payload.pop("message_id", None)
                result = self._post("sendMessage", json=payload)
            if result:
                self._remember_message(chat_id, result)
                with self._state_lock:
                    self._message_ids[chat_id] = int(result["message_id"])

    def start_updates(self, status: str) -> None:
        if not self.enabled:
            return
        self.stop_updates()
        with self._state_lock:
            self._status = self._query_status = status
            self._started = time.monotonic()
            self._message_ids.clear()
            self._sent_message_ids.clear()
            self._command_offset = None
        self._stop.clear()
        self._poll_commands(timeout=0)
        self._publish_status()
        self._thread = threading.Thread(target=self._update_loop, name="telegram-pipeline-updates", daemon=True)
        self._thread.start()

    def update(self, status: str) -> None:
        with self._state_lock:
            self._status = self._query_status = status

    def track(self, status: str) -> None:
        with self._state_lock:
            self._query_status = status

    @staticmethod
    def _progress_status(status: str, current: int, total: int, started: float) -> str:
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = current / elapsed
        eta = max(total - current, 0) / rate if rate else 0.0
        return (
            f"{status}\nProgress: {current:,}/{total:,} ({current / total:.1%}) | "
            f"{rate:,.1f} it/s | ETA: {eta:.0f}s"
        )

    def progress(self, status: str, current: int, total: int, started: float) -> None:
        self.update(self._progress_status(status, current, total, started))

    def track_progress(self, status: str, current: int, total: int, started: float) -> None:
        self.track(self._progress_status(status, current, total, started))

    def _handle_command(self, chat_id: int, message_id: int, text: str) -> None:
        if chat_id not in self._chat_ids or not text.startswith("/"):
            return
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in {"/progress", "/status"}:
            self._send_message(chat_id, self._status_text(query=True))
        elif command == "/help":
            self._send_message(
                chat_id,
                "Pipeline commands:\n/progress - current progress\n/status - current status\n"
                "/clear - delete this run's bot messages\n/help - list commands",
            )
        elif command == "/clear":
            with self._state_lock:
                message_ids = set(self._sent_message_ids.pop(chat_id, set()))
                self._message_ids.pop(chat_id, None)
            message_ids.add(message_id)
            for sent_message_id in sorted(message_ids):
                self._post(
                    "deleteMessage",
                    json={"chat_id": chat_id, "message_id": sent_message_id},
                )
            self._publish_status((chat_id,))

    def _handle_callback(
        self,
        chat_id: int,
        message_id: int,
        callback_id: str,
        data: str,
    ) -> None:
        command = data.removeprefix("pipeline:")
        if chat_id not in self._chat_ids or command not in {"progress", "status", "help", "clear"}:
            return
        self._post("answerCallbackQuery", json={"callback_query_id": callback_id})
        self._handle_command(chat_id, message_id, f"/{command}")

    def _poll_commands(self, *, timeout: int = 1) -> None:
        offset = self._command_offset
        result = self._post(
            "getUpdates",
            json={
                "offset": -1 if offset is None else offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
        )
        if not isinstance(result, list):
            return
        update_ids = [update.get("update_id") for update in result if isinstance(update, dict)]
        update_ids = [int(update_id) for update_id in update_ids if isinstance(update_id, int)]
        previous = -1 if offset is None else offset - 1
        self._command_offset = max(update_ids, default=previous) + 1
        if offset is None:
            return
        for update in result:
            message = update.get("message") if isinstance(update, dict) else None
            chat = message.get("chat") if isinstance(message, dict) else None
            text = message.get("text") if isinstance(message, dict) else None
            message_id = message.get("message_id") if isinstance(message, dict) else None
            chat_id = chat.get("id") if isinstance(chat, dict) else None
            if isinstance(chat_id, int) and isinstance(message_id, int) and isinstance(text, str):
                self._handle_command(chat_id, message_id, text.strip())
            callback = update.get("callback_query") if isinstance(update, dict) else None
            callback_message = callback.get("message") if isinstance(callback, dict) else None
            callback_chat = (
                callback_message.get("chat") if isinstance(callback_message, dict) else None
            )
            callback_id = callback.get("id") if isinstance(callback, dict) else None
            callback_data = callback.get("data") if isinstance(callback, dict) else None
            callback_message_id = (
                callback_message.get("message_id")
                if isinstance(callback_message, dict)
                else None
            )
            callback_chat_id = (
                callback_chat.get("id") if isinstance(callback_chat, dict) else None
            )
            if (
                isinstance(callback_chat_id, int)
                and isinstance(callback_message_id, int)
                and isinstance(callback_id, str)
                and isinstance(callback_data, str)
            ):
                self._handle_callback(
                    callback_chat_id,
                    callback_message_id,
                    callback_id,
                    callback_data,
                )

    def _update_loop(self) -> None:
        last_publish = time.monotonic()
        while not self._stop.wait(0.25):
            self._poll_commands()
            if time.monotonic() - last_publish >= self._update_interval:
                self._publish_status()
                last_publish = time.monotonic()

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
        detail_block = f"\n<pre>{html.escape(detail)}</pre>" if detail else ""
        self.notify_html(
            f"<b>{html.escape(phase.upper())}</b>\n<code>{html.escape(symbol)}</code>{detail_block}"
        )

    def notify_failure(self, error: BaseException, elapsed: float) -> None:
        detail = _redacted_error(error)
        self.stop_updates(f"Pipeline failed | {elapsed:.1f}s | {detail}")
        self.notify_html(f"<b>PIPELINE ERROR</b>\n<pre>{html.escape(detail)}</pre>")

    def _send_file(self, method: str, field: str, path: str | Path, caption: str) -> None:
        source = Path(path)
        if not self.enabled or not source.is_file():
            return
        for chat_id in self._chat_ids:
            try:
                with source.open("rb") as stream:
                    result = self._post(
                        method,
                        data={"chat_id": chat_id, "caption": caption[:1024]},
                        files={field: (source.name, stream)},
                    )
                    self._remember_message(chat_id, result)
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
