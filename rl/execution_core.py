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
        self._decision_index = decision_bars.index
        self._m1_index = m1_bars.index
        self._m1_bounds = self._m1_index.searchsorted(self._decision_index, side="right")
        self._m1_open = m1_bars["Open"].to_numpy(dtype=float, copy=False)
        self._m1_high = m1_bars["High"].to_numpy(dtype=float, copy=False)
        self._m1_low = m1_bars["Low"].to_numpy(dtype=float, copy=False)
        self._m1_close = m1_bars["Close"].to_numpy(dtype=float, copy=False)
        self._decision_close = decision_bars["Close"].to_numpy(dtype=float, copy=False)
        self._decision_atr = decision_bars["atr"].to_numpy(dtype=float, copy=False) if "atr" in decision_bars else None
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

    def _open(
        self,
        direction: int,
        risk_fraction: float,
        sl_atr_multiplier: float,
        tp_sl_ratio: float,
        timestamp: pd.Timestamp,
        raw_price: float,
        atr: float,
        spread: float,
        slippage: float,
    ) -> None:
        entry = self._entry_price(raw_price, direction, spread, slippage)
        stop_distance = max(sl_atr_multiplier * atr, 1e-12)
        risk_cash = max(self.equity * risk_fraction, 1e-12)
        quantity = risk_cash / stop_distance
        commission = entry * quantity * self.commission_rate
        self.equity -= commission
        self.position = SimulatedPosition(
            direction=direction,
            entry_time=timestamp,
            entry_price=entry,
            stop_price=entry - direction * stop_distance,
            take_profit_price=entry + direction * stop_distance * tp_sl_ratio,
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

    def _scan_position(self, start: int, stop: int, spread: float, slippage: float) -> float:
        position = self.position
        if not position.direction or start >= stop:
            return 0.0
        if position.direction == 1:
            stop_hits = self._m1_low[start:stop] <= position.stop_price
            target_hits = self._m1_high[start:stop] >= position.take_profit_price
        else:
            stop_hits = self._m1_high[start:stop] >= position.stop_price
            target_hits = self._m1_low[start:stop] <= position.take_profit_price
        hits = np.flatnonzero(stop_hits | target_hits)
        if not len(hits):
            return 0.0
        offset = int(hits[0])
        stopped = bool(stop_hits[offset])
        return self._close(
            self._m1_index[start + offset],
            position.stop_price if stopped else position.take_profit_price,
            "stop" if stopped else "take_profit",
            spread,
            slippage,
        )

    def execute_interval(
        self,
        decision_index: int,
        intent: TradeIntent,
        *,
        spread: float = 0.0,
        slippage: float = 0.0,
        latency_ticks: int = 1,
    ) -> StepResult:
        return StepResult(
            *self._execute_values(
                decision_index,
                intent.direction,
                float(intent.risk_fraction),
                float(intent.sl_atr_multiplier),
                float(intent.tp_sl_ratio),
                spread,
                slippage,
                latency_ticks,
            )
        )

    def execute_values(
        self,
        decision_index: int,
        direction: int,
        risk_fraction: float,
        sl_atr_multiplier: float,
        tp_sl_ratio: float,
        *,
        spread: float = 0.0,
        slippage: float = 0.0,
        latency_ticks: int = 1,
    ) -> tuple[float, float, float]:
        reward, realized_r, _, _, _, equity = self._execute_values(
            decision_index,
            direction,
            risk_fraction,
            sl_atr_multiplier,
            tp_sl_ratio,
            spread,
            slippage,
            latency_ticks,
        )
        return reward, realized_r, equity

    def _execute_values(
        self,
        decision_index: int,
        direction: int,
        risk_fraction: float,
        sl_atr_multiplier: float,
        tp_sl_ratio: float,
        spread: float,
        slippage: float,
        latency_ticks: int,
    ) -> tuple[float, float, float, float, float, float]:
        if not 0 <= decision_index < len(self.decision_bars) - 1 or latency_ticks < 1:
            raise ValueError("invalid decision index or latency")
        start, end = self._decision_index[decision_index : decision_index + 2]
        lo, hi = int(self._m1_bounds[decision_index]), int(self._m1_bounds[decision_index + 1])
        action_tick = lo + latency_ticks - 1
        atr_value = max(
            float(self._decision_atr[decision_index])
            if self._decision_atr is not None
            else float(self.decision_bars.iloc[decision_index]["atr"]),
            1e-12,
        )
        split = min(action_tick, hi)
        realized_r = self._scan_position(lo, split, spread, slippage)
        if action_tick < hi:
            timestamp = self._m1_index[action_tick]
            if direction != self.position.direction:
                if self.position.direction:
                    realized_r += self._close(timestamp, self._m1_open[action_tick], "signal", spread, slippage)
                if direction:
                    self._open(
                        direction,
                        risk_fraction,
                        sl_atr_multiplier,
                        tp_sl_ratio,
                        timestamp,
                        self._m1_open[action_tick],
                        atr_value,
                        spread,
                        slippage,
                    )
            realized_r += self._scan_position(action_tick, hi, spread, slippage)

        holding_cost = 0.0
        if self.position.direction:
            self.position.decision_bars += 1
            holding_cost = self.holding_cost_r
            self.equity -= self.position.risk_cash * holding_cost
            if self.position.decision_bars >= self.max_holding_bars:
                raw_exit = self._m1_close[hi - 1] if hi > lo else self._decision_close[decision_index + 1]
                realized_r += self._close(end, raw_exit, "timeout", spread, slippage)

        differential = self._differential_sharpe(realized_r)
        penalty = self.downside_penalty * min(realized_r, 0) ** 2
        reward = realized_r + self.differential_sharpe_weight * differential - penalty - holding_cost
        return reward, realized_r, differential, penalty, holding_cost, self.equity
