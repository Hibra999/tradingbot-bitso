from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
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
    PaperExecutionEngine,
    RiskManager,
    TrackedPosition,
    TradeIntent,
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


class _PartialExitREST:
    def __init__(self):
        self.events = []
        self.exit_origin = None

    async def cancel_order(self, order_id):
        self.events.append(("cancel", order_id))

    async def place_order(self, payload):
        if "stop" in payload:
            self.events.append(("protect", payload["major"]))
            return {"oid": "replacement-stop"}
        self.exit_origin = payload["origin_id"]
        self.events.append(("exit", payload["major"]))
        return {"oid": "partial-exit"}

    async def request(self, method, path, *, params=None, private=False, **kwargs):
        amount = "0" if params["origin_id"] == "old-stop-origin" else "0.4"
        self.events.append(("reconcile", params["origin_id"]))
        return [] if amount == "0" else [{"major": amount, "price": "100", "fees_amount": "0"}]


class _PreflightREST:
    async def request(self, method, path, **kwargs):
        if path == "/available_books":
            rule = _rules()
            return [{key: str(value) if isinstance(value, Decimal) else value for key, value in rule.__dict__.items()}]
        if path == "/fees":
            return {"fees": [{"book": "btc_usd"}]}
        if path == "/balance":
            return {"balances": [{"currency": "btc", "available": "1"}, {"currency": "usd", "available": "1000"}]}
        if path == "/open_orders":
            return []
        raise AssertionError(path)


def _rules(*, margin: bool = False) -> BookRules:
    return BookRules(
        "btc_usd",
        Decimal("0.1"),
        Decimal("0.001"),
        Decimal("100"),
        Decimal("1"),
        Decimal("1000000"),
        Decimal("1"),
        Decimal("1000000"),
        margin,
    )


def _intent(direction: int = 1) -> TradeIntent:
    return TradeIntent(
        direction, Decimal("0.005"), Decimal("1"), Decimal("1"), 0.8, (0.1, 0.2, 0.7), "model", "btc_usd", datetime.now(timezone.utc)
    )


class ExecutionEngineTests(unittest.IsolatedAsyncioTestCase):
    def test_risk_requires_tick_limits_and_confirmed_margin(self) -> None:
        risk = RiskManager(RuntimeRiskParams(), allow_margin_shorts=True, margin_capability_confirmed=True)
        with self.assertRaisesRegex(ValueError, "tick"):
            risk.validate_intent(_intent(), _rules(), Decimal("100.01"), Decimal("1"))
        with self.assertRaises(PermissionError):
            risk.validate_intent(_intent(-1), _rules(), Decimal("100"), Decimal("1"))
        risk.validate_intent(_intent(-1), replace(_rules(), margin_enabled=True), Decimal("100"), Decimal("1"))

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
            self.assertTrue(journal.get_state("kill_latch")["latched"])
            self.assertEqual(recovered.cancelled, ["stop"])
            journal.close()

    async def test_synthetic_tp_cancels_and_reconciles_stop_before_exit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            rest = _RaceREST()
            engine = LiveExecutionEngine(
                journal,
                rest,
                {
                    "schema_version": 2,
                    "selected_artifact": "model",
                    "artifact_bundle": {"action_contract": "long_flat_spot"},
                },
            )
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

    async def test_partial_exit_rearms_exchange_stop(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            rest = _PartialExitREST()
            engine = LiveExecutionEngine(
                journal,
                rest,
                {
                    "schema_version": 2,
                    "selected_artifact": "model",
                    "artifact_bundle": {"action_contract": "long_flat_spot"},
                },
            )
            engine.enabled = True
            engine.rules = {"btc_usd": _rules()}
            engine.position = TrackedPosition(
                "btc_usd", 1, Decimal("1"), Decimal("100"), Decimal("90"), Decimal("110"), "entry", "old-stop-origin"
            )
            engine.bracket = Bracket("btc_usd", "entry-order", "old-stop", Decimal("110"), Decimal("90"), Decimal("1"))
            engine.open_order_ids = {"old-stop"}
            self.assertTrue(await engine.trigger_take_profit(Decimal("111")))
            self.assertEqual(engine.position.quantity, Decimal("0.6"))
            self.assertEqual(engine.bracket.stop_order_id, "replacement-stop")
            self.assertEqual(engine.bracket.quantity, Decimal("0.6"))
            self.assertEqual(rest.events[:3], [("cancel", "old-stop"), ("reconcile", "old-stop-origin"), ("exit", "1")])
            self.assertEqual(rest.events[-1], ("protect", "0.6"))
            journal.close()

    async def test_preflight_never_clears_a_persisted_kill_latch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            journal.set_state("kill_latch", {"latched": True})
            engine = LiveExecutionEngine(
                journal,
                _PreflightREST(),
                {
                    "schema_version": 2,
                    "selected_artifact": "model",
                    "artifact_bundle": {"action_contract": "long_flat_spot"},
                },
            )
            await engine.preflight(("btc_usd",))
            self.assertTrue(engine.frozen)
            self.assertFalse(engine.enabled)
            journal.close()

    async def test_paper_bracket_consumes_live_depth(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            journal = ExecutionJournal(Path(folder) / "journal.sqlite3")
            book = LocalOrderBook("btc_usd")
            book.bootstrap({"sequence": "1", "bids": [{"oid": "b", "price": "99", "amount": "100"}], "asks": [{"oid": "a", "price": "100", "amount": "100"}]})
            engine = PaperExecutionEngine(journal, {"btc_usd": book}, {"btc_usd": _rules()})
            await engine.execute(_intent(), Decimal("10"))
            self.assertIsNotNone(engine.bracket)
            book.bootstrap({"sequence": "2", "bids": [{"oid": "b", "price": "89", "amount": "100"}], "asks": [{"oid": "a", "price": "90", "amount": "100"}]})
            self.assertTrue(await engine.trigger_bracket(Decimal("89")))
            self.assertTrue(engine.is_flat)
            self.assertEqual(journal.events()[-1]["payload"]["reason"], "stop")
            journal.close()


if __name__ == "__main__":
    unittest.main()
