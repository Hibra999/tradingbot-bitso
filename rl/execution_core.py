from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np, pandas as pd
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
        self._m1_o = m1_bars["Open"].to_numpy(dtype=float)
        self._m1_h = m1_bars["High"].to_numpy(dtype=float)
        self._m1_l = m1_bars["Low"].to_numpy(dtype=float)
        self._m1_c = m1_bars["Close"].to_numpy(dtype=float)
        self._m1_idx = m1_bars.index
        self._dec_idx = decision_bars.index
        self._dec_c = decision_bars["Close"].to_numpy(dtype=float)
        self._dec_atr = decision_bars["atr"].to_numpy(dtype=float) if "atr" in decision_bars else None
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
        stop_dist = max(float(intent.sl_atr_multiplier) * atr, 1e-12)
        risk_cash = max(self.equity * float(intent.risk_fraction), 1e-12)
        quantity = risk_cash / stop_dist
        commission = entry * quantity * self.commission_rate
        self.equity -= commission
        self.position = SimulatedPosition(
            direction=direction,
            entry_time=timestamp,
            entry_price=entry,
            stop_price=entry - direction * stop_dist,
            take_profit_price=entry + direction * stop_dist * float(intent.tp_sl_ratio),
            quantity=quantity,
            risk_cash=risk_cash,
            entry_commission=commission,
        )

    def _close(self, timestamp: pd.Timestamp, raw_price: float, reason: str, spread: float, slippage: float) -> float:
        pos = self.position
        exit_p = self._exit_price(raw_price, pos.direction, spread, slippage)
        commission = exit_p * pos.quantity * self.commission_rate
        exit_pnl = (exit_p - pos.entry_price) * pos.quantity * pos.direction - commission
        pnl = exit_pnl - pos.entry_commission
        self.equity += exit_pnl
        realized_r = pnl / max(pos.risk_cash, 1e-12)
        self.trades.append({
            "entry_time": pos.entry_time,
            "exit_time": timestamp,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_p,
            "stop_price": pos.stop_price,
            "take_profit_price": pos.take_profit_price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "r_multiple": realized_r,
            "reason": reason,
        })
        self.position = SimulatedPosition()
        return realized_r

    def _differential_sharpe(self, realized_r: float, eta: float = 0.01) -> float:
        m, sq = self.return_mean, self.return_square_mean
        var = sq - m ** 2
        val = (sq * (realized_r - m) - 0.5 * m * (realized_r ** 2 - sq)) / (var ** 1.5) if var > 1e-12 else 0.0
        self.return_mean = m + eta * (realized_r - m)
        self.return_square_mean = sq + eta * (realized_r ** 2 - sq)
        return float(val)

    def execute_interval(self, decision_index: int, intent: TradeIntent, *, spread: float = 0.0, slippage: float = 0.0, latency_ticks: int = 1) -> StepResult:
        if not 0 <= decision_index < len(self.decision_bars) - 1 or latency_ticks < 1: raise ValueError("invalid decision index or latency")
        start, end = self._dec_idx[decision_index], self._dec_idx[decision_index + 1]
        lo = int(self._m1_idx.searchsorted(start, side="right"))
        hi = int(self._m1_idx.searchsorted(end, side="right"))
        action_tick = lo + latency_ticks - 1
        realized_r = 0.0
        atr_val = max(float(self._dec_atr[decision_index]) if self._dec_atr is not None else float(self.decision_bars.iloc[decision_index]["atr"]), 1e-12)
        m1_o, m1_h, m1_l, m1_idx = self._m1_o, self._m1_h, self._m1_l, self._m1_idx

        for tick in range(lo, hi):
            ts, op = m1_idx[tick], m1_o[tick]
            if tick == action_tick and intent.direction != self.position.direction:
                if self.position.direction: realized_r += self._close(ts, op, "signal", spread, slippage)
                if intent.direction: self._open(intent, ts, op, atr_val, spread, slippage)
            pos = self.position
            if not pos.direction: continue
            if pos.direction == 1:
                stop_hit = m1_l[tick] <= pos.stop_price
                target_hit = m1_h[tick] >= pos.take_profit_price
            else:
                stop_hit = m1_h[tick] >= pos.stop_price
                target_hit = m1_l[tick] <= pos.take_profit_price
            if stop_hit: realized_r += self._close(ts, pos.stop_price, "stop", spread, slippage)
            elif target_hit: realized_r += self._close(ts, pos.take_profit_price, "take_profit", spread, slippage)

        holding_cost = 0.0
        if self.position.direction:
            self.position.decision_bars += 1
            holding_cost = self.holding_cost_r
            self.equity -= self.position.risk_cash * holding_cost
            if self.position.decision_bars >= self.max_holding_bars:
                raw_exit = float(self._m1_c[hi - 1]) if hi > lo else float(self._dec_c[decision_index + 1])
                realized_r += self._close(end, raw_exit, "timeout", spread, slippage)

        diff = self._differential_sharpe(realized_r)
        penalty = self.downside_penalty * min(realized_r, 0) ** 2
        return StepResult(realized_r + self.differential_sharpe_weight * diff - penalty - holding_cost, realized_r, diff, penalty, holding_cost, self.equity)
