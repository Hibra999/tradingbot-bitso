from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import RuntimeRiskParams
from execution import BaseExecutionEngine, ExecutionJournal, RiskManager
from telegram_bot import BacktestManager, TelegramService, parse_allowed_chat_ids
from ui import DashboardController


class _Engine(BaseExecutionEngine):
    def __init__(self, journal):
        super().__init__(journal, RiskManager(RuntimeRiskParams()), "paper")

    async def cancel_order(self, order_id: str) -> None:
        return None

    async def liquidate(self) -> bool:
        return True


class _Message:
    def __init__(self):
        self.replies = []

    async def reply_text(self, value: str) -> None:
        self.replies.append(value)


class TelegramTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.journal = ExecutionJournal(Path(self.folder.name) / "journal.sqlite3")
        self.controller = DashboardController({"paper": _Engine(self.journal)})

    async def asyncTearDown(self) -> None:
        self.journal.close()
        self.folder.cleanup()

    async def test_chat_allowlist_and_alert_secret_redaction(self) -> None:
        self.assertEqual(parse_allowed_chat_ids("1, 2"), frozenset({1, 2}))
        with self.assertRaises(ValueError):
            parse_allowed_chat_ids("anyone")
        with patch.dict(os.environ, {"BITSO_API_SECRET": "private-value"}, clear=False):
            service = TelegramService(
                self.controller,
                token="123456:valid-test-token",
                allowed_chat_ids=frozenset({1}),
                queue_size=2,
            )
        calls = []

        async def handler(update, context):
            calls.append(update.effective_chat.id)

        guarded = service._authorized(handler)
        unauthorized = SimpleNamespace(effective_chat=SimpleNamespace(id=9), effective_message=_Message())
        authorized = SimpleNamespace(effective_chat=SimpleNamespace(id=1), effective_message=_Message())
        await guarded(unauthorized, SimpleNamespace(args=[]))
        await guarded(authorized, SimpleNamespace(args=[]))
        self.assertEqual(calls, [1])
        service.alert_nowait("123456:valid-test-token private-value")
        alert = await service.alerts.get()
        self.assertNotIn("valid-test-token", alert)
        self.assertNotIn("private-value", alert)

    async def test_backtest_parameters_are_whitelisted_before_spawn(self) -> None:
        manager = BacktestManager(Path(__file__).resolve().parents[1], timeout_seconds=1)
        with self.assertRaises(ValueError):
            await manager.run({"profile": "smoke", "symbol": "DOGE/USD"})


if __name__ == "__main__":
    unittest.main()
