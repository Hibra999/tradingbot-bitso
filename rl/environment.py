from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from validation import DomainRandomizer

from .actions import ppo_intent, qrdqn_intent, sac_intent
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
    ):
        super().__init__()
        self.decision_bars = decision_bars
        self.m1_bars = m1_bars
        self.feature_columns = feature_columns
        self.action_mode = action_mode
        self.book = book
        self.model_id = model_id
        self.randomize = randomize
        self.base_spread = base_spread
        self.core = BracketExecutionCore(decision_bars, m1_bars)
        self._timestamps = decision_bars.index.to_pydatetime()
        self.action_space = {
            "ppo": spaces.MultiDiscrete([3, 4, 4]),
            "sac": spaces.Box(
                low=np.array([-1, 0.005, 1, 1], dtype=np.float32),
                high=np.array([1, 0.03, 3.5, 4], dtype=np.float32),
            ),
            "qrdqn": spaces.Discrete(129),
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
        if self.randomize:
            randomized = DomainRandomizer(seed or 0).perturb(features, self.decision_bars["atr"], self.base_spread)
            self.episode_features = randomized.features
            self.spreads, self.slippages, self.latencies = randomized.spread, randomized.slippage, randomized.latency_ticks
        else:
            self.episode_features = features
            self.spreads = np.full(len(features), self.base_spread)
            self.slippages = np.zeros(len(features))
            self.latencies = np.ones(len(features), dtype=int)
        self._feature_values = np.nan_to_num(
            self.episode_features.to_numpy(dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0
        )
        return self._observation(), {}

    def step(self, action):
        timestamp = self._timestamps[self.index]
        kwargs = {"model_id": self.model_id, "book": self.book, "timestamp": timestamp}
        if self.action_mode == "ppo":
            intent = ppo_intent(action, **kwargs)
        elif self.action_mode == "sac":
            intent = sac_intent(action, **kwargs)
        else:
            intent = qrdqn_intent(int(action), **kwargs)
        result = self.core.execute_interval(
            self.index,
            intent,
            spread=float(self.spreads[self.index]),
            slippage=float(self.slippages[self.index]),
            latency_ticks=int(self.latencies[self.index]),
        )
        self.index += 1
        terminated = self.index >= len(self.decision_bars) - 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._observation()
        return observation, result.reward, terminated, False, {"equity": result.equity, "realized_r": result.realized_r}
