from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from functools import lru_cache

import numpy as np

from config import RLConfig
from execution import TradeIntent


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
) -> TradeIntent:
    cfg = config or RLConfig()
    direction_index, sl_index, tp_index = (int(value) for value in action)
    if direction_index not in (0, 1, 2):
        raise ValueError("PPO direction must be 0=flat, 1=long, or 2=short")
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
) -> TradeIntent:
    direction_score, risk, sl, tp = (float(value) for value in action)
    if not (-1 <= direction_score <= 1 and 0.005 <= risk <= 0.03 and 1 <= sl <= 3.5 and 1 <= tp <= 4):
        raise ValueError("SAC action is outside its declared Box")
    direction = 0 if abs(direction_score) < 0.1 else int(np.sign(direction_score))
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


@lru_cache(maxsize=16)
def _qrdqn_action_table(
    risk_fractions: tuple[float, ...],
    sl_atr_multipliers: tuple[float, ...],
    tp_sl_ratios: tuple[float, ...],
) -> tuple[tuple[int, float, float, float], ...]:
    actions: list[tuple[int, float, float, float]] = [
        (0, risk_fractions[0], sl_atr_multipliers[0], tp_sl_ratios[0])
    ]
    actions.extend(
        (direction, risk, sl, tp)
        for direction in (-1, 1)
        for risk in risk_fractions
        for sl in sl_atr_multipliers
        for tp in tp_sl_ratios
    )
    if len(actions) != 129:
        raise ValueError("QR-DQN action table must contain exactly 129 actions")
    return tuple(actions)


def qrdqn_action_table(config: RLConfig | None = None) -> tuple[tuple[int, float, float, float], ...]:
    cfg = config or RLConfig()
    return _qrdqn_action_table(cfg.risk_fractions, cfg.sl_atr_multipliers, cfg.tp_sl_ratios)


def qrdqn_intent(
    action: int,
    *,
    model_id: str,
    book: str,
    timestamp: datetime,
    scores: tuple[float, ...] = (),
    config: RLConfig | None = None,
) -> TradeIntent:
    direction, risk, sl, tp = qrdqn_action_table(config)[int(action)]
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
