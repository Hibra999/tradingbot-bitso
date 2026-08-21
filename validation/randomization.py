from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationConfig:
    spread_lognormal_sigma: float = 0.15
    slippage_atr_fraction: float = 0.02
    feature_noise_sigma: float = 0.01
    min_latency_ticks: int = 1
    max_latency_ticks: int = 3


@dataclass(frozen=True)
class RandomizedEpisode:
    features: pd.DataFrame
    spread: np.ndarray
    slippage: np.ndarray
    latency_ticks: np.ndarray


class DomainRandomizer:
    def __init__(self, seed: int, config: PerturbationConfig | None = None):
        self.seed = int(seed)
        self.config = config or PerturbationConfig()
        self.rng = np.random.default_rng(self.seed)

    def perturb(
        self,
        features: pd.DataFrame,
        atr: pd.Series,
        base_spread: float | np.ndarray | pd.Series,
    ) -> RandomizedEpisode:
        spread_base = np.asarray(base_spread, dtype=float)
        if spread_base.ndim == 0:
            spread_base = np.full(len(features), float(spread_base))
        if len(features) != len(atr) or spread_base.shape != (len(features),) or bool((spread_base < 0).any()):
            raise ValueError("features/ATR must align and base_spread must be non-negative")
        cfg = self.config
        spread = spread_base * self.rng.lognormal(
            mean=-0.5 * cfg.spread_lognormal_sigma**2,
            sigma=cfg.spread_lognormal_sigma,
            size=len(features),
        )
        slippage = atr.to_numpy(dtype=float) * self.rng.uniform(0, cfg.slippage_atr_fraction, len(features))
        noisy = features.astype(float) + self.rng.normal(0, cfg.feature_noise_sigma, features.shape)
        latency = self.rng.integers(cfg.min_latency_ticks, cfg.max_latency_ticks + 1, len(features))
        return RandomizedEpisode(noisy, spread, slippage, latency)
