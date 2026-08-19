from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


class SequenceGapError(RuntimeError):
    def __init__(self, expected: int, received: int):
        super().__init__(f"order-book sequence gap: expected {expected}, received {received}")
        self.expected = expected
        self.received = received


@dataclass(frozen=True)
class BookOrder:
    order_id: str
    side: str
    price: Decimal
    amount: Decimal


class LocalOrderBook:
    def __init__(self, book: str):
        self.book = book
        self.sequence: int | None = None
        self.orders: dict[str, BookOrder] = {}

    def bootstrap(self, snapshot: dict) -> None:
        orders: dict[str, BookOrder] = {}
        for side in ("bids", "asks"):
            for item in snapshot.get(side, []):
                order_id = str(item["oid"])
                orders[order_id] = BookOrder(order_id, side, Decimal(item["price"]), Decimal(item["amount"]))
        self.orders = orders
        self.sequence = int(snapshot["sequence"])

    def apply_diff(self, message: dict) -> bool:
        sequence = int(message["sequence"])
        if self.sequence is None:
            raise RuntimeError("order book is not bootstrapped")
        if sequence <= self.sequence:
            return False
        if sequence != self.sequence + 1:
            raise SequenceGapError(self.sequence + 1, sequence)
        for change in message.get("payload", []):
            order_id = str(change["o"])
            if change.get("s") != "open" or "a" not in change:
                self.orders.pop(order_id, None)
            else:
                side = "bids" if int(change["t"]) == 0 else "asks"
                self.orders[order_id] = BookOrder(order_id, side, Decimal(change["r"]), Decimal(change["a"]))
        self.sequence = sequence
        return True

    def levels(self, side: str) -> list[tuple[Decimal, Decimal]]:
        if side not in {"bids", "asks"}:
            raise ValueError("side must be bids or asks")
        totals: dict[Decimal, Decimal] = {}
        for order in self.orders.values():
            if order.side == side:
                totals[order.price] = totals.get(order.price, Decimal("0")) + order.amount
        return sorted(totals.items(), reverse=side == "bids")
