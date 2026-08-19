from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

Direction = Literal[-1, 0, 1]


@dataclass(frozen=True)
class TradeIntent:
    direction: Direction
    risk_fraction: Decimal
    sl_atr_multiplier: Decimal
    tp_sl_ratio: Decimal
    confidence: float
    action_distribution: tuple[float, ...]
    model_id: str
    book: str
    timestamp: datetime


@dataclass(frozen=True)
class OrderTicket:
    order_id: str
    origin_id: str
    book: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop"]
    quantity: Decimal
    price: Decimal | None = None
    status: str = "pending"
    created_at: datetime | None = None


@dataclass(frozen=True)
class Fill:
    order_id: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime


@dataclass(frozen=True)
class Bracket:
    book: str
    entry_order_id: str
    stop_order_id: str | None
    take_profit_price: Decimal
    stop_price: Decimal
    quantity: Decimal
    status: str = "open"


@dataclass(frozen=True)
class Balance:
    currency: str
    available: Decimal
    locked: Decimal = Decimal("0")


@dataclass(frozen=True)
class EngineSnapshot:
    mode: Literal["paper", "live"]
    state: str
    equity: Decimal
    realized_pnl: Decimal
    balances: tuple[Balance, ...] = field(default_factory=tuple)
    open_orders: tuple[OrderTicket, ...] = field(default_factory=tuple)
    brackets: tuple[Bracket, ...] = field(default_factory=tuple)
    updated_at: datetime | None = None
