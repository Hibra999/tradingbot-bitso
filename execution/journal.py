from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

_SECRET_WORDS = ("secret", "password", "token", "authorization", "api_key")


def safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if any(word in str(key).lower() for word in _SECRET_WORDS) else safe_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [safe_payload(item) for item in value]
    if isinstance(value, (Decimal, datetime, Path)):
        return str(value)
    return value


class ExecutionJournal:
    """Small durable event/state journal; one process owns one instance."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    def append(self, event_type: str, payload: dict[str, Any]) -> int:
        encoded = json.dumps(safe_payload(payload), sort_keys=True, separators=(",", ":"))
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO events(created_at, event_type, payload) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), event_type, encoded),
            )
        return int(cursor.lastrowid)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        encoded = json.dumps(safe_payload(value), sort_keys=True, separators=(",", ":"))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO state(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, encoded, now),
            )

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def events(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, created_at, event_type, payload FROM events WHERE id > ? ORDER BY id",
                (after_id,),
            ).fetchall()
        return [
            {"id": row[0], "created_at": row[1], "event_type": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "ExecutionJournal":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
