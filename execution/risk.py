from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from config import RuntimeRiskParams

from .models import TradeIntent


@dataclass(frozen=True)
class BookRules:
    book: str
    tick_size: Decimal
    minimum_amount: Decimal
    maximum_amount: Decimal
    minimum_price: Decimal
    maximum_price: Decimal
    minimum_value: Decimal
    maximum_value: Decimal
    margin_enabled: bool

    @classmethod
    def from_payload(cls, payload: dict) -> "BookRules":
        return cls(
            book=payload["book"],
            tick_size=Decimal(payload["tick_size"]),
            minimum_amount=Decimal(payload["minimum_amount"]),
            maximum_amount=Decimal(payload["maximum_amount"]),
            minimum_price=Decimal(payload["minimum_price"]),
            maximum_price=Decimal(payload["maximum_price"]),
            minimum_value=Decimal(payload["minimum_value"]),
            maximum_value=Decimal(payload["maximum_value"]),
            margin_enabled=bool(payload.get("margin_enabled")),
        )


class RiskManager:
    def __init__(
        self,
        params: RuntimeRiskParams,
        *,
        allow_margin_shorts: bool = False,
        margin_capability_confirmed: bool = False,
    ):
        self.params = params
        self.allow_margin_shorts = allow_margin_shorts
        self.margin_capability_confirmed = margin_capability_confirmed

    def validate_intent(self, intent: TradeIntent, rules: BookRules, price: Decimal, quantity: Decimal) -> None:
        if intent.book != rules.book:
            raise ValueError("intent book does not match validated rules")
        if not Decimal("0") <= intent.risk_fraction <= self.params.risk_fraction:
            raise ValueError("intent exceeds runtime risk fraction")
        if intent.direction == -1 and not (
            self.allow_margin_shorts and self.margin_capability_confirmed and rules.margin_enabled
        ):
            raise PermissionError("shorting requires flags plus confirmed account/book margin capability")
        value = price * quantity
        if not rules.minimum_amount <= quantity <= rules.maximum_amount:
            raise ValueError("quantity violates book limits")
        if not rules.minimum_price <= price <= rules.maximum_price:
            raise ValueError("price violates book limits")
        if not rules.minimum_value <= value <= rules.maximum_value:
            raise ValueError("order value violates book limits")
        if value > self.params.max_position_usd:
            raise ValueError("order value exceeds runtime position limit")
