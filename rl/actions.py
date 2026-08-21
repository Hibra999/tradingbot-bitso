from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from functools import lru_cache

import numpy as np

from config import RLConfig
from execution import TradeIntent

_SAC_LOW = (-1.0, float(np.float32(0.005)), 1.0, 1.0)
_SAC_HIGH = (1.0, 0.03, 3.5, 4.0)


def _sac_action_values(action: np.ndarray | list[float] | tuple[float, float, float, float]) -> tuple[float, ...]:
    direction, risk, stop, target = (float(value) for value in action)
    if not (
        _SAC_LOW[0] <= direction <= _SAC_HIGH[0]
        and _SAC_LOW[1] <= risk <= _SAC_HIGH[1]
        and _SAC_LOW[2] <= stop <= _SAC_HIGH[2]
        and _SAC_LOW[3] <= target <= _SAC_HIGH[3]
    ):
        raise ValueError("SAC action is outside its declared Box")
    return (
        min(1.0, max(-1.0, direction)),
        min(0.03, max(0.005, risk)),
        min(3.5, max(1.0, stop)),
        min(4.0, max(1.0, target)),
    )


def _intent(
    direction: int,
    risk: float,
    sl: float,
    tp: float,
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    confidence: float,
    distribution: tuple[float, ...],
) -> TradeIntent:
    return TradeIntent(
        direction=direction,  # type: ignore[arg-type]
        risk_fraction=Decimal(str(risk)),
        sl_atr_multiplier=Decimal(str(sl)),
        tp_sl_ratio=Decimal(str(tp)),
        confidence=float(np.clip(confidence, 0, 1)),
        action_distribution=distribution,
        model_id=model_id,
        book=book,
        timestamp=timestamp,
    )


def ppo_intent(
    action: np.ndarray | list[int] | tuple[int, int, int],
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    distribution: tuple[float, ...] = (),
    config: RLConfig | None = None,
    allow_short: bool = True,
) -> TradeIntent:
    cfg = config or RLConfig()
    direction_index, sl_index, tp_index = (int(value) for value in action)
    allowed = (0, 1, 2) if allow_short else (0, 1)
    if direction_index not in allowed:
        raise ValueError("PPO direction is unavailable for the configured action contract")
    direction = (0, 1, -1)[direction_index]
    confidence = max(distribution) if distribution else 1.0
    return _intent(
        direction,
        cfg.risk_fractions[0],
        cfg.sl_atr_multipliers[sl_index],
        cfg.tp_sl_ratios[tp_index],
        model_id=model_id,
        book=book,
        timestamp=timestamp,
        confidence=confidence,
        distribution=distribution,
    )


def sac_intent(
    action: np.ndarray | list[float] | tuple[float, float, float, float],
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    allow_short: bool = True,
) -> TradeIntent:
    direction_score, risk, sl, tp = _sac_action_values(action)
    direction = 0 if abs(direction_score) < 0.1 else int(np.sign(direction_score))
    if not allow_short and direction < 0:
        direction = 0
    derived = (max(-direction_score, 0), max(1 - abs(direction_score), 0), max(direction_score, 0))
    return _intent(
        direction,
        risk,
        sl,
        tp,
        model_id=model_id,
        book=book,
        timestamp=timestamp,
        confidence=abs(direction_score),
        distribution=tuple(float(value) for value in derived),
    )


def target_exposure_intent(
    action: np.ndarray | list[float] | tuple[float] | float,
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    max_risk_fraction: float = 0.005,
    no_trade_band: float = 0.10,
) -> TradeIntent:
    values = np.asarray(action, dtype=float).reshape(-1)
    if values.shape != (1,) or not np.isfinite(values[0]) or not 0 <= values[0] <= 1:
        raise ValueError("target exposure is outside its declared Box")
    target = float(values[0])
    if not 0 <= no_trade_band < 1 or not 0 < max_risk_fraction <= 0.03:
        raise ValueError("invalid target-exposure intent contract")
    if target < no_trade_band:
        target = 0.0
    return _intent(
        int(target > 0),
        max_risk_fraction * target,
        2.0,
        2.0,
        model_id=model_id,
        book=book,
        timestamp=timestamp,
        confidence=target,
        distribution=(1.0 - target, target),
    )


@lru_cache(maxsize=16)
def _qrdqn_action_table(
    risk_fractions: tuple[float, ...],
    sl_atr_multipliers: tuple[float, ...],
    tp_sl_ratios: tuple[float, ...],
    allow_short: bool,
) -> tuple[tuple[int, float, float, float], ...]:
    actions: list[tuple[int, float, float, float]] = [
        (0, risk_fractions[0], sl_atr_multipliers[0], tp_sl_ratios[0])
    ]
    actions.extend(
        (direction, risk, sl, tp)
        for direction in ((-1, 1) if allow_short else (1,))
        for risk in risk_fractions
        for sl in sl_atr_multipliers
        for tp in tp_sl_ratios
    )
    expected = 129 if allow_short else 65
    if len(actions) != expected:
        raise ValueError(f"QR-DQN action table must contain exactly {expected} actions")
    return tuple(actions)


def qrdqn_action_table(
    config: RLConfig | None = None,
    *,
    allow_short: bool = True,
) -> tuple[tuple[int, float, float, float], ...]:
    cfg = config or RLConfig()
    return _qrdqn_action_table(
        cfg.risk_fractions,
        cfg.sl_atr_multipliers,
        cfg.tp_sl_ratios,
        allow_short,
    )


def qrdqn_intent(
    action: int,
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    scores: tuple[float, ...] = (),
    config: RLConfig | None = None,
    allow_short: bool = True,
) -> TradeIntent:
    direction, risk, sl, tp = qrdqn_action_table(config, allow_short=allow_short)[int(action)]
    confidence = max(scores) if scores else 1.0
    return _intent(
        direction,
        risk,
        sl,
        tp,
        model_id=model_id,
        book=book,
        timestamp=timestamp,
        confidence=confidence,
        distribution=scores,
    )
