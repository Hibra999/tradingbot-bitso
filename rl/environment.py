from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import RLConfig
from validation import DomainRandomizer, PerturbationConfig

from .actions import _sac_action_values, qrdqn_action_table
from .execution_core import BracketExecutionCore


class BracketTradingEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        decision_bars: pd.DataFrame,
        m1_bars: pd.DataFrame,
        feature_columns: list[str],
        *,
        action_mode: Literal["ppo", "sac", "qrdqn"] = "ppo",
        book: str = "btc_usd",
        model_id: str = "training",
        randomize: bool = True,
        base_spread: float = 0.0,
        base_spread_bps: float = 2.0,
        commission_rate: float = 0.001,
        perturbation_config: PerturbationConfig | None = None,
        random_seed: int = 0,
        allow_short: bool = False,
    ):
        super().__init__()
        if base_spread < 0 or base_spread_bps < 0 or commission_rate < 0:
            raise ValueError("commission and spread assumptions must be non-negative")
        self.decision_bars = decision_bars
        self.m1_bars = m1_bars
        self.feature_columns = feature_columns
        self.action_mode = action_mode
        self.book = book
        self.model_id = model_id
        self.randomize = randomize
        self.base_spread = base_spread
        self.base_spread_bps = base_spread_bps
        self.perturbation_config = perturbation_config
        self.allow_short = allow_short
        self._random_seed = int(random_seed)
        self.core = BracketExecutionCore(decision_bars, m1_bars, commission_rate=commission_rate)
        cfg = RLConfig()
        self._risk_fraction = cfg.risk_fractions[0]
        self._sl_atr_multipliers, self._tp_sl_ratios = cfg.sl_atr_multipliers, cfg.tp_sl_ratios
        self._qrdqn_actions = qrdqn_action_table(cfg, allow_short=allow_short)
        self.action_space = {
            "ppo": spaces.MultiDiscrete([3 if allow_short else 2, 4, 4]),
            "sac": spaces.Box(
                low=np.array([-1, 0.005, 1, 1], dtype=np.float32),
                high=np.array([1, 0.03, 3.5, 4], dtype=np.float32),
            ),
            "qrdqn": spaces.Discrete(len(self._qrdqn_actions)),
        }[action_mode]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(feature_columns) + 3,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        position = self.core.position
        observation = np.empty(self.observation_space.shape, dtype=np.float32)
        observation[:-3] = self._feature_values[self.index]
        observation[-3:] = position.direction, position.decision_bars / 24, self.core.equity / self.core.initial_equity - 1
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        self.core.reset()  # every disjoint CPCV segment starts flat
        features = self.decision_bars[self.feature_columns]
        base_spreads = (
            np.full(len(features), self.base_spread)
            if self.base_spread > 0
            else self.decision_bars["Close"].to_numpy(dtype=float) * self.base_spread_bps / 10_000
        )
        if self.randomize:
            if seed is not None:
                self._random_seed = int(seed)
            randomized = DomainRandomizer(self._random_seed, self.perturbation_config).perturb(
                features, self.decision_bars["atr"], base_spreads
            )
            self.episode_features = randomized.features
            self.spreads, self.slippages, self.latencies = randomized.spread, randomized.slippage, randomized.latency_ticks
        else:
            self.episode_features = features
            self.spreads = base_spreads
            self.slippages = np.zeros(len(features))
            self.latencies = np.ones(len(features), dtype=int)
        self._feature_values = np.nan_to_num(
            self.episode_features.to_numpy(dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0
        )
        return self._observation(), {}

    def step(self, action):
        if self.action_mode == "ppo":
            direction_index, sl_index, tp_index = (int(value) for value in action)
            allowed = (0, 1, 2) if self.allow_short else (0, 1)
            if direction_index not in allowed:
                raise ValueError("PPO direction is unavailable for the configured action space")
            direction, risk = (0, 1, -1)[direction_index], self._risk_fraction
            sl, tp = self._sl_atr_multipliers[sl_index], self._tp_sl_ratios[tp_index]
        elif self.action_mode == "sac":
            direction_score, risk, sl, tp = _sac_action_values(action)
            direction = 0 if direction_score < 0.1 else 1
            if self.allow_short and direction_score <= -0.1:
                direction = -1
        else:
            direction, risk, sl, tp = self._qrdqn_actions[int(action)]
        reward, realized_r, equity = self.core.execute_values(
            self.index,
            int(direction),
            float(risk),
            float(sl),
            float(tp),
            spread=float(self.spreads[self.index]),
            slippage=float(self.slippages[self.index]),
            latency_ticks=int(self.latencies[self.index]),
        )
        self.index += 1
        truncated = self.index >= len(self.decision_bars) - 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if truncated else self._observation()
        return observation, reward, False, truncated, {"equity": equity, "realized_r": realized_r}
