from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from execution import TradeIntent


@dataclass
class SimulatedPosition:
    direction: int = 0
    entry_time: pd.Timestamp | None = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    quantity: float = 0.0
    risk_cash: float = 0.0
    entry_commission: float = 0.0
    decision_bars: int = 0


@dataclass(frozen=True)
class StepResult:
    reward: float
    realized_r: float
    differential_sharpe: float
    downside_penalty: float
    holding_cost_r: float
    equity: float


class BracketExecutionCore:
    def __init__(
        self,
        decision_bars: pd.DataFrame,
        m1_bars: pd.DataFrame,
        *,
        initial_equity: float = 10_000.0,
        commission_rate: float = 0.001,
        holding_cost_r: float = 0.001,
        downside_penalty: float = 0.1,
        differential_sharpe_weight: float = 0.05,
        max_holding_bars: int = 24,
    ):
        if not decision_bars.index.is_monotonic_increasing or not m1_bars.index.is_monotonic_increasing:
            raise ValueError("decision and M1 bars must be chronological")
        self.decision_bars = decision_bars
        self.m1_bars = m1_bars
        self.initial_equity = initial_equity
        self.commission_rate = commission_rate
        self.holding_cost_r = holding_cost_r
        self.downside_penalty = downside_penalty
        self.differential_sharpe_weight = differential_sharpe_weight
        self.max_holding_bars = max_holding_bars
        self.reset()

    def reset(self) -> None:
        self.equity = float(self.initial_equity)
        self.position = SimulatedPosition()
        self.trades: list[dict[str, Any]] = []
        self.return_mean = 0.0
        self.return_square_mean = 0.0

    @staticmethod
    def _entry_price(price: float, direction: int, spread: float, slippage: float) -> float:
        return price + direction * (spread / 2 + slippage)

    @staticmethod
    def _exit_price(price: float, direction: int, spread: float, slippage: float) -> float:
        return price - direction * (spread / 2 + slippage)

    def _open(self, intent: TradeIntent, timestamp: pd.Timestamp, raw_price: float, atr: float, spread: float, slippage: float) -> None:
        direction = intent.direction
        entry = self._entry_price(raw_price, direction, spread, slippage)
        stop_distance = max(float(intent.sl_atr_multiplier) * atr, 1e-12)
        risk_cash = max(self.equity * float(intent.risk_fraction), 1e-12)
        quantity = risk_cash / stop_distance
        commission = entry * quantity * self.commission_rate
        self.equity -= commission
        self.position = SimulatedPosition(
            direction=direction,
            entry_time=timestamp,
            entry_price=entry,
            stop_price=entry - direction * stop_distance,
            take_profit_price=entry + direction * stop_distance * float(intent.tp_sl_ratio),
            quantity=quantity,
            risk_cash=risk_cash,
            entry_commission=commission,
        )

    def _close(self, timestamp: pd.Timestamp, raw_price: float, reason: str, spread: float, slippage: float) -> float:
        position = self.position
        exit_price = self._exit_price(raw_price, position.direction, spread, slippage)
        commission = exit_price * position.quantity * self.commission_rate
        exit_pnl = (exit_price - position.entry_price) * position.quantity * position.direction - commission
        pnl = exit_pnl - position.entry_commission
        self.equity += exit_pnl
        realized_r = pnl / max(position.risk_cash, 1e-12)
        self.trades.append(
            {
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "stop_price": position.stop_price,
                "take_profit_price": position.take_profit_price,
                "quantity": position.quantity,
                "pnl": pnl,
                "r_multiple": realized_r,
                "reason": reason,
            }
        )
        self.position = SimulatedPosition()
        return realized_r

    def _differential_sharpe(self, realized_r: float, eta: float = 0.01) -> float:
        mean, square_mean = self.return_mean, self.return_square_mean
        variance = square_mean - mean**2
        value = 0.0
        if variance > 1e-12:
            value = (square_mean * (realized_r - mean) - 0.5 * mean * (realized_r**2 - square_mean)) / variance**1.5
        self.return_mean = mean + eta * (realized_r - mean)
        self.return_square_mean = square_mean + eta * (realized_r**2 - square_mean)
        return float(value)

    def execute_interval(
        self,
        decision_index: int,
        intent: TradeIntent,
        *,
        spread: float = 0.0,
        slippage: float = 0.0,
        latency_ticks: int = 1,
    ) -> StepResult:
        if not 0 <= decision_index < len(self.decision_bars) - 1 or latency_ticks < 1:
            raise ValueError("invalid decision index or latency")
        start, end = self.decision_bars.index[decision_index : decision_index + 2]
        lo = int(self.m1_bars.index.searchsorted(start, side="right"))
        hi = int(self.m1_bars.index.searchsorted(end, side="right"))
        action_tick = lo + latency_ticks - 1
        realized_r = 0.0
        atr_value = max(float(self.decision_bars.iloc[decision_index]["atr"]), 1e-12)

        for tick in range(lo, hi):
            row = self.m1_bars.iloc[tick]
            timestamp = self.m1_bars.index[tick]
            if tick == action_tick and intent.direction != self.position.direction:
                if self.position.direction:
                    realized_r += self._close(timestamp, float(row["Open"]), "signal", spread, slippage)
                if intent.direction:
                    self._open(intent, timestamp, float(row["Open"]), atr_value, spread, slippage)

            position = self.position
            if not position.direction:
                continue
            if position.direction == 1:
                stop_hit = float(row["Low"]) <= position.stop_price
                target_hit = float(row["High"]) >= position.take_profit_price
            else:
                stop_hit = float(row["High"]) >= position.stop_price
                target_hit = float(row["Low"]) <= position.take_profit_price
            if stop_hit:  # pessimistic when both prices trade inside one M1 bar
                realized_r += self._close(timestamp, position.stop_price, "stop", spread, slippage)
            elif target_hit:
                realized_r += self._close(timestamp, position.take_profit_price, "take_profit", spread, slippage)

        holding_cost = 0.0
        if self.position.direction:
            self.position.decision_bars += 1
            holding_cost = self.holding_cost_r
            self.equity -= self.position.risk_cash * holding_cost
            if self.position.decision_bars >= self.max_holding_bars:
                raw_exit = float(self.m1_bars.iloc[hi - 1]["Close"]) if hi > lo else float(self.decision_bars.iloc[decision_index + 1]["Close"])
                realized_r += self._close(end, raw_exit, "timeout", spread, slippage)

        differential = self._differential_sharpe(realized_r)
        penalty = self.downside_penalty * min(realized_r, 0) ** 2
        reward = realized_r + self.differential_sharpe_weight * differential - penalty - holding_cost
        return StepResult(reward, realized_r, differential, penalty, holding_cost, self.equity)
