from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from config import RuntimeRiskParams
from execution import (
    BaseExecutionEngine,
    Bracket,
    BookRules,
    ExecutionJournal,
    InsufficientDepthError,
    KillSwitch,
    LiveExecutionEngine,
    LocalOrderBook,
    RiskManager,
    TrackedPosition,
    consume_depth,
)


class _RaceREST:
    def __init__(self):
        self.events = []
        self.exit_origin = None

    async def cancel_order(self, order_id):
        self.events.append(("cancel", order_id))

    async def place_order(self, payload):
        self.exit_origin = payload["origin_id"]
        self.events.append(("exit", payload["major"]))
        return {"oid": "exit-order"}

    async def request(self, method, path, *, params=None, private=False, **kwargs):
        origin = params["origin_id"]
        amount = "0.4" if origin == "stop-origin" else "0.6"
        self.events.append(("reconcile", origin))
        return [{"major": amount, "price": "100", "fees_amount": "0"}]


class _KillEngine(BaseExecutionEngine):
    def __init__(self, journal):
        super().__init__(journal, RiskManager(RuntimeRiskParams()), "paper")
        if self.position is None:
            self.position = TrackedPosition("btc_usd", 1, Decimal("1"), Decimal("100"), Decimal("90"), Decimal("110"), "entry")
            self.open_order_ids = {"stop"}
        self.cancelled = []

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        self.open_order_ids.discard(order_id)

    async def liquidate(self) -> bool:
        self.position = None
        return True


class ExecutionEngineTests(unittest.IsolatedAsyncioTestCase):
    def test_depth_is_pessimistic_and_rejects_shortfall(self) -> None:
        book = LocalOrderBook("btc_usd")
        book.bootstrap(
            {
                "sequence": "1",
                "bids": [{"oid": "b", "price": "99", "amount": "5"}],
                "asks": [
                    {"oid": "a1", "price": "100", "amount": "1"},
                    {"oid": "a2", "price": "101", "amount": "2"},
                ],
            }
        )
        fill = consume_depth(book, "buy", Decimal("2"))
        self.assertEqual(fill.average_price, Decimal("100.5"))
        with self.assertRaises(InsufficientDepthError):
            consume_depth(book, "buy", Decimal("4"))

    async def test_restart_recovery_and_idempotent_kill_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            engine = _KillEngine(journal)
            engine.persist()
            recovered = _KillEngine(journal)
            self.assertEqual(recovered.position.quantity, Decimal("1"))

            kill = KillSwitch(recovered)
            first, second = await kill.trigger("test"), await kill.trigger("again")
            self.assertEqual(first, second)
            self.assertLess(first.dispatch_latency_ms, 500)
            self.assertTrue(first.confirmed_flat)
            self.assertTrue(recovered.frozen)
            self.assertEqual(recovered.cancelled, ["stop"])
            journal.close()

    async def test_synthetic_tp_cancels_and_reconciles_stop_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            rest = _RaceREST()
            engine = LiveExecutionEngine(journal, rest, {"schema_version": 1, "selected_artifact": "model"})
            engine.enabled = True
            engine.position = TrackedPosition(
                "btc_usd",
                1,
                Decimal("1"),
                Decimal("100"),
                Decimal("90"),
                Decimal("110"),
                "entry-origin",
                "stop-origin",
            )
            engine.bracket = Bracket(
                "btc_usd", "entry-order", "stop-order", Decimal("110"), Decimal("90"), Decimal("1")
            )
            engine.open_order_ids = {"stop-order"}
            self.assertTrue(await engine.trigger_take_profit(Decimal("111")))
            self.assertTrue(engine.is_flat)
            self.assertEqual(rest.events[0], ("cancel", "stop-order"))
            self.assertEqual(rest.events[1], ("reconcile", "stop-origin"))
            self.assertEqual(rest.events[2], ("exit", "0.6"))
            journal.close()


if __name__ == "__main__":
    unittest.main()
