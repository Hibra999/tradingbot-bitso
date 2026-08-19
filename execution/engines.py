from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Literal

from config import RuntimeRiskParams

from .bitso_rest import BitsoRESTClient
from .journal import ExecutionJournal
from .models import Bracket, TradeIntent
from .order_book import LocalOrderBook
from .risk import BookRules, RiskManager


class InsufficientDepthError(RuntimeError):
    pass


@dataclass(frozen=True)
class DepthFill:
    quantity: Decimal
    average_price: Decimal
    notional: Decimal


@dataclass
class TrackedPosition:
    book: str
    direction: int
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    entry_origin_id: str
    stop_origin_id: str | None = None

    def payload(self) -> dict[str, str | int | None]:
        return {key: str(value) if isinstance(value, Decimal) else value for key, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrackedPosition":
        return cls(
            book=payload["book"],
            direction=int(payload["direction"]),
            quantity=Decimal(payload["quantity"]),
            entry_price=Decimal(payload["entry_price"]),
            stop_price=Decimal(payload["stop_price"]),
            take_profit_price=Decimal(payload["take_profit_price"]),
            entry_origin_id=payload["entry_origin_id"],
            stop_origin_id=payload.get("stop_origin_id"),
        )


def consume_depth(book: LocalOrderBook, side: Literal["buy", "sell"], quantity: Decimal) -> DepthFill:
    levels = book.levels("asks" if side == "buy" else "bids")
    remaining, notional = quantity, Decimal("0")
    for price, available in levels:
        consumed = min(remaining, available)
        notional += consumed * price
        remaining -= consumed
        if remaining == 0:
            return DepthFill(quantity, notional / quantity, notional)
    raise InsufficientDepthError(f"insufficient {side} depth for {quantity}")


class BaseExecutionEngine:
    def __init__(self, journal: ExecutionJournal, risk: RiskManager, mode: Literal["paper", "live"]):
        self.journal = journal
        self.risk = risk
        self.mode = mode
        self.frozen = False
        self.position: TrackedPosition | None = None
        self.bracket: Bracket | None = None
        self.open_order_ids: set[str] = set()
        self._order_lock = asyncio.Lock()
        self.recover()

    def recover(self) -> None:
        state = self.journal.get_state("engine")
        if not state:
            return
        self.frozen = bool(state.get("frozen", True))
        self.open_order_ids = set(state.get("open_order_ids", []))
        if state.get("position"):
            self.position = TrackedPosition.from_payload(state["position"])
        if state.get("bracket"):
            payload = state["bracket"]
            self.bracket = Bracket(
                book=payload["book"],
                entry_order_id=payload["entry_order_id"],
                stop_order_id=payload.get("stop_order_id"),
                take_profit_price=Decimal(payload["take_profit_price"]),
                stop_price=Decimal(payload["stop_price"]),
                quantity=Decimal(payload["quantity"]),
                status=payload["status"],
            )

    def persist(self) -> None:
        bracket = None
        if self.bracket:
            bracket = {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in asdict(self.bracket).items()
            }
        self.journal.set_state(
            "engine",
            {
                "mode": self.mode,
                "frozen": self.frozen,
                "position": self.position.payload() if self.position else None,
                "bracket": bracket,
                "open_order_ids": sorted(self.open_order_ids),
            },
        )

    @property
    def is_flat(self) -> bool:
        return self.position is None or self.position.quantity <= 0

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    async def liquidate(self) -> bool:
        raise NotImplementedError


class PaperExecutionEngine(BaseExecutionEngine):
    def __init__(
        self,
        journal: ExecutionJournal,
        books: dict[str, LocalOrderBook],
        rules: dict[str, BookRules],
        *,
        equity: Decimal = Decimal("10000"),
        fee_rate: Decimal = Decimal("0.001"),
        risk_params: RuntimeRiskParams | None = None,
    ):
        super().__init__(journal, RiskManager(risk_params or RuntimeRiskParams()), "paper")
        self.books, self.rules, self.equity, self.fee_rate = books, rules, equity, fee_rate

    async def execute(self, intent: TradeIntent, atr: Decimal) -> DepthFill | None:
        async with self._order_lock:
            if self.frozen:
                raise PermissionError("engine is frozen")
            if intent.direction == -1:
                raise PermissionError("paper spot is long/flat by default")
            if intent.direction == 0:
                return await self._close_position()
            if self.position:
                return None
            levels = self.books[intent.book].levels("asks")
            if not levels:
                raise InsufficientDepthError("empty ask book")
            reference = levels[0][0]
            stop_distance = intent.sl_atr_multiplier * atr
            risk_cash = self.equity * intent.risk_fraction
            quantity = min(risk_cash / stop_distance, self.risk.params.max_position_usd / reference)
            self.risk.validate_intent(intent, self.rules[intent.book], reference, quantity)
            fill = consume_depth(self.books[intent.book], "buy", quantity)
            fee = fill.notional * self.fee_rate
            self.equity -= fill.notional + fee
            origin = uuid.uuid4().hex
            self.position = TrackedPosition(
                intent.book,
                1,
                fill.quantity,
                fill.average_price,
                fill.average_price - stop_distance,
                fill.average_price + stop_distance * intent.tp_sl_ratio,
                origin,
            )
            self.journal.append("paper_entry", {"book": intent.book, "quantity": fill.quantity, "price": fill.average_price, "fee": fee})
            self.persist()
            return fill

    async def _close_position(self) -> DepthFill | None:
        if not self.position:
            return None
        fill = consume_depth(self.books[self.position.book], "sell", self.position.quantity)
        self.equity += fill.notional * (Decimal("1") - self.fee_rate)
        self.journal.append("paper_exit", {"book": self.position.book, "quantity": fill.quantity, "price": fill.average_price})
        self.position = self.bracket = None
        self.persist()
        return fill

    async def cancel_order(self, order_id: str) -> None:
        self.open_order_ids.discard(order_id)

    async def liquidate(self) -> bool:
        try:
            await self._close_position()
        except InsufficientDepthError:
            return False
        return self.is_flat


class LiveExecutionEngine(BaseExecutionEngine):
    def __init__(
        self,
        journal: ExecutionJournal,
        rest: BitsoRESTClient,
        approved_manifest: dict[str, Any],
        *,
        risk_params: RuntimeRiskParams | None = None,
        allow_margin_shorts: bool = False,
        margin_capability_confirmed: bool = False,
    ):
        risk = RiskManager(
            risk_params or RuntimeRiskParams(),
            allow_margin_shorts=allow_margin_shorts,
            margin_capability_confirmed=margin_capability_confirmed,
        )
        super().__init__(journal, risk, "live")
        self.rest = rest
        self.approved_manifest = approved_manifest
        self.rules: dict[str, BookRules] = {}
        self.enabled = False

    async def preflight(self, books: tuple[str, ...]) -> None:
        available, fees, balances, open_orders = await asyncio.gather(
            self.rest.request("GET", "/available_books"),
            self.rest.request("GET", "/fees", private=True),
            self.rest.request("GET", "/balance", private=True),
            self.rest.request("GET", "/open_orders", private=True),
        )
        by_book = {item["book"]: item for item in available}
        if any(book not in by_book for book in books):
            raise RuntimeError("configured Bitso book is unavailable")
        self.rules = {book: BookRules.from_payload(by_book[book]) for book in books}
        if any(rule.tick_size <= 0 for rule in self.rules.values()):
            raise RuntimeError("invalid book tick size")
        fee_books = {item["book"] for item in fees.get("fees", [])}
        if not set(books) <= fee_books:
            raise RuntimeError("account fee tier is missing for a configured book")
        currencies = {item["currency"] for item in balances.get("balances", [])}
        if not {part for book in books for part in book.split("_")} <= currencies:
            raise RuntimeError("required balances are missing")
        self.journal.set_state(
            "balances",
            {
                item["currency"]: {"available": item.get("available", "0"), "locked": item.get("locked", "0")}
                for item in balances.get("balances", [])
            },
        )
        unknown_orders = [item for item in open_orders if item.get("oid") not in self.open_order_ids]
        if unknown_orders:
            raise RuntimeError("unmanaged existing orders must be reconciled before startup")
        if self.approved_manifest.get("schema_version") != 1 or not self.approved_manifest.get("selected_artifact"):
            raise RuntimeError("approved model manifest schema is invalid")
        self.enabled = True
        self.frozen = False
        self.persist()

    async def _filled(self, origin_id: str) -> tuple[Decimal, Decimal, Decimal]:
        trades = await self.rest.request("GET", "/order_trades", params={"origin_id": origin_id}, private=True)
        quantity = sum((Decimal(item["major"]) for item in trades), Decimal("0"))
        notional = sum((Decimal(item["major"]) * Decimal(item["price"]) for item in trades), Decimal("0"))
        fees = sum((Decimal(item.get("fees_amount", "0")) for item in trades), Decimal("0"))
        return quantity, (notional / quantity if quantity else Decimal("0")), fees

    async def execute(self, intent: TradeIntent, atr: Decimal, reference_price: Decimal) -> Any:
        async with self._order_lock:
            if not self.enabled or self.frozen:
                raise PermissionError("live decisions are not enabled")
            if intent.direction == 0:
                return await self._exit_position("signal")
            if self.position:
                return None
            stop_distance = intent.sl_atr_multiplier * atr
            quantity = min(
                self.risk.params.max_position_usd / reference_price,
                self.risk.params.max_position_usd * intent.risk_fraction / stop_distance,
            )
            self.risk.validate_intent(intent, self.rules[intent.book], reference_price, quantity)
            side = "buy" if intent.direction == 1 else "sell"
            origin = uuid.uuid4().hex
            payload = {"book": intent.book, "side": side, "type": "market", "major": str(quantity), "origin_id": origin}
            if intent.direction == -1:
                payload["margin_order_type"] = "CROSS_MARGIN"
            order = await self.rest.place_order(payload)
            entry_order_id = order.get("oid", origin)
            filled, entry_price, _ = await self._filled(origin)
            if filled <= 0:
                self.open_order_ids.add(entry_order_id)
                self.persist()
                return order
            stop_price = entry_price - intent.direction * stop_distance
            take_profit = entry_price + intent.direction * stop_distance * intent.tp_sl_ratio
            stop_origin = uuid.uuid4().hex
            stop_side = "sell" if intent.direction == 1 else "buy"
            stop_payload = {
                "book": intent.book,
                "side": stop_side,
                "type": "market",
                "major": str(filled),
                "stop": str(stop_price),
                "origin_id": stop_origin,
            }
            if intent.direction == -1:
                stop_payload["margin_order_type"] = "CROSS_MARGIN"
            stop_order = await self.rest.place_order(stop_payload)
            stop_id = stop_order.get("oid", stop_origin)
            self.position = TrackedPosition(intent.book, intent.direction, filled, entry_price, stop_price, take_profit, origin, stop_origin)
            self.bracket = Bracket(intent.book, entry_order_id, stop_id, take_profit, stop_price, filled)
            self.open_order_ids.add(stop_id)
            self.journal.append("live_entry", {"book": intent.book, "quantity": filled, "entry_order_id": entry_order_id, "stop_order_id": stop_id})
            self.persist()
            return order

    async def _cancel_stop_and_reconcile(self) -> Decimal:
        if not self.position or not self.bracket or not self.bracket.stop_order_id:
            return Decimal("0")
        stop_id = self.bracket.stop_order_id
        await self.rest.cancel_order(stop_id)
        self.open_order_ids.discard(stop_id)
        filled, _, _ = await self._filled(self.position.stop_origin_id or stop_id)
        return min(filled, self.position.quantity)

    async def _exit_position(self, reason: str) -> Any:
        if not self.position:
            return None
        stop_filled = await self._cancel_stop_and_reconcile()
        self.position.quantity -= stop_filled
        if self.position.quantity <= 0:
            self.position = self.bracket = None
            self.persist()
            return None
        origin = uuid.uuid4().hex
        side = "sell" if self.position.direction == 1 else "buy"
        payload = {
            "book": self.position.book,
            "side": side,
            "type": "market",
            "major": str(self.position.quantity),
            "origin_id": origin,
        }
        if self.position.direction == -1:
            payload["margin_order_type"] = "CROSS_MARGIN"
        order = await self.rest.place_order(payload)
        filled, _, _ = await self._filled(origin)
        self.position.quantity -= min(filled, self.position.quantity)
        self.journal.append("live_exit", {"book": self.position.book, "filled": filled, "reason": reason})
        if self.position.quantity <= 0:
            self.position = self.bracket = None
        self.persist()
        return order

    async def trigger_take_profit(self, executable_price: Decimal) -> bool:
        async with self._order_lock:
            if not self.position:
                return False
            hit = executable_price >= self.position.take_profit_price if self.position.direction == 1 else executable_price <= self.position.take_profit_price
            if not hit:
                return False
            await self._exit_position("synthetic_take_profit")
            return True

    async def cancel_order(self, order_id: str) -> None:
        await self.rest.cancel_order(order_id)
        self.open_order_ids.discard(order_id)
        self.persist()

    async def liquidate(self) -> bool:
        async with self._order_lock:
            await self._exit_position("kill")
            return self.is_flat


@dataclass(frozen=True)
class KillReport:
    dispatch_latency_ms: float
    exchange_ack_latency_ms: float
    confirmed_flat_latency_ms: float | None
    confirmed_flat: bool


class KillSwitch:
    def __init__(self, engine: BaseExecutionEngine, *, retries: int = 5):
        self.engine = engine
        self.retries = retries
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[KillReport] | None = None

    async def trigger(self, reason: str) -> KillReport:
        async with self._lock:
            if self._task is None:
                self._task = asyncio.create_task(self._run(reason))
            task = self._task
        return await asyncio.shield(task)

    async def _run(self, reason: str) -> KillReport:
        started = time.perf_counter()
        self.engine.frozen = True
        self.engine.journal.set_state("kill_latch", {"latched": True})
        self.engine.journal.append("kill_triggered", {"reason": reason})
        self.engine.persist()
        cancellations = [asyncio.create_task(self.engine.cancel_order(order_id)) for order_id in tuple(self.engine.open_order_ids)]
        dispatch_latency = (time.perf_counter() - started) * 1_000
        if cancellations:
            await asyncio.gather(*cancellations, return_exceptions=True)
        acknowledgement_latency = (time.perf_counter() - started) * 1_000
        for attempt in range(self.retries):
            if await self.engine.liquidate():
                break
            await asyncio.sleep(min(2**attempt, 8))
        flat = self.engine.is_flat
        flat_latency = (time.perf_counter() - started) * 1_000 if flat else None
        report = KillReport(dispatch_latency, acknowledgement_latency, flat_latency, flat)
        self.engine.journal.append("kill_complete", asdict(report))
        self.engine.persist()
        return report
