from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pufferlib import PufferEnv

from config import RLConfig
from quant import ALPHA_TARGET_COLUMN
from validation import DomainRandomizer, PerturbationConfig

from .actions import TARGET_EXPOSURE_LEVELS
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
        reward_baseline_column: str | None = None,
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
        if reward_baseline_column and reward_baseline_column not in decision_bars:
            raise ValueError("reward baseline column is missing from decision bars")
        self.reward_baseline_column = reward_baseline_column
        self._random_seed = int(random_seed)
        self.core = BracketExecutionCore(decision_bars, m1_bars, commission_rate=commission_rate)
        self._baseline_core = (
            BracketExecutionCore(decision_bars, m1_bars, commission_rate=commission_rate)
            if reward_baseline_column
            else None
        )
        cfg = RLConfig()
        self._max_risk_fraction = cfg.risk_fractions[0]
        self._discrete_exposures = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        self.action_space = {
            "ppo": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "sac": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "qrdqn": spaces.Discrete(len(self._discrete_exposures)),
        }[action_mode]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(feature_columns) + 7,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        observation = np.empty(self.observation_space.shape, dtype=np.float32)
        observation[:-7] = self._feature_values[self.index]
        observation[-7:] = self.core.state_values(
            self.index,
            spread=float(self.spreads[self.index]),
            slippage=float(self.slippages[self.index]),
        )
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        self.core.reset()  # every disjoint CPCV segment starts flat
        if self._baseline_core:
            self._baseline_core.reset()
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

    def baseline_target_exposure(self) -> float:
        if not self.reward_baseline_column:
            raise RuntimeError("environment has no reward baseline")
        return float(
            np.clip(
                self.episode_features[self.reward_baseline_column].iloc[self.index],
                0.0,
                1.0,
            )
        )

    def step(self, action):
        if self.action_mode in {"ppo", "sac"}:
            values = np.asarray(action, dtype=float).reshape(-1)
            if values.shape != (1,) or not np.isfinite(values[0]) or not 0 <= values[0] <= 1:
                raise ValueError("continuous target exposure is outside its declared Box")
            target_exposure = float(values[0])
        else:
            action_index = int(np.asarray(action).item())
            if not 0 <= action_index < len(self._discrete_exposures):
                raise ValueError("discrete target exposure is outside its declared action space")
            target_exposure = float(self._discrete_exposures[action_index])
        return self.step_target(target_exposure)

    def step_target(self, target_exposure: float):
        reward, realized_r, equity = self.core.execute_target_exposure(
            self.index,
            target_exposure,
            max_risk_fraction=self._max_risk_fraction,
            spread=float(self.spreads[self.index]),
            slippage=float(self.slippages[self.index]),
            latency_ticks=int(self.latencies[self.index]),
        )
        if self._baseline_core:
            baseline_reward, _, _ = self._baseline_core.execute_target_exposure(
                self.index,
                self.baseline_target_exposure(),
                max_risk_fraction=self._max_risk_fraction,
                spread=float(self.spreads[self.index]),
                slippage=float(self.slippages[self.index]),
                latency_ticks=int(self.latencies[self.index]),
            )
            reward -= baseline_reward
        self.index += 1
        truncated = self.index >= len(self.decision_bars) - 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if truncated else self._observation()
        return observation, reward, False, truncated, {
            "equity": equity,
            "realized_r": realized_r,
            "target_exposure": self.core.target_exposure,
        }


class PufferTradingEnv(PufferEnv):
    def __init__(
        self,
        decision_segments: tuple[pd.DataFrame, ...],
        m1_bars: pd.DataFrame,
        feature_columns: list[str],
        *,
        num_agents: int,
        episode_steps: int,
        random_seed: int,
        commission_rate: float,
        base_spread_bps: float,
        perturbation_config: PerturbationConfig,
    ):
        if not decision_segments or min(num_agents, episode_steps) < 1:
            raise ValueError("Puffer training requires segments, agents, and episode steps")
        if ALPHA_TARGET_COLUMN not in feature_columns:
            raise ValueError("Puffer residual training requires causal alpha exposure")
        self._decision_segments = decision_segments
        self._m1_bars = m1_bars
        self._feature_columns = feature_columns
        self._episode_steps = episode_steps
        self._commission_rate = commission_rate
        self._base_spread_bps = base_spread_bps
        self._perturbation_config = perturbation_config
        self._window_counts = tuple(
            len(segment) - episode_steps for segment in decision_segments
        )
        for count in self._window_counts:
            if count < 1:
                raise ValueError(
                    f"Puffer training segments need at least {episode_steps + 1} H1 bars"
                )
        self.envs = [self._new_environment(random_seed + index) for index in range(num_agents)]
        self.num_agents = num_agents
        self.agents_per_batch = num_agents
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = spaces.Discrete(len(TARGET_EXPOSURE_LEVELS))
        self._base_seed = int(random_seed)
        self._episodes = np.zeros(num_agents, dtype=np.int64)
        self._episode_returns = np.zeros(num_agents, dtype=np.float64)
        super().__init__()

    def _new_environment(self, seed: int) -> BracketTradingEnvV2:
        window = int(np.random.default_rng(seed).integers(sum(self._window_counts)))
        for segment, count in zip(self._decision_segments, self._window_counts):
            if window < count:
                decision_bars = segment.iloc[window : window + self._episode_steps + 1]
                break
            window -= count
        else:
            raise RuntimeError("Puffer episode window selection failed")
        return BracketTradingEnvV2(
            decision_bars,
            self._m1_bars,
            self._feature_columns,
            action_mode="ppo",
            randomize=True,
            random_seed=seed,
            allow_short=False,
            commission_rate=self._commission_rate,
            base_spread_bps=self._base_spread_bps,
            perturbation_config=self._perturbation_config,
            reward_baseline_column=ALPHA_TARGET_COLUMN,
        )

    def _reset_agent(self, index: int) -> None:
        seed = self._base_seed + index + int(self._episodes[index]) * self.num_agents
        self.envs[index].close()
        self.envs[index] = self._new_environment(seed)
        observation, _ = self.envs[index].reset(seed=seed)
        self.observations[index] = observation
        self._episode_returns[index] = 0.0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._base_seed = int(seed)
        self._episodes.fill(0)
        self.rewards.fill(0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        self.masks.fill(True)
        for index in range(self.num_agents):
            self._reset_agent(index)
        return self.observations, []

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if (
            actions.shape != (self.num_agents,)
            or not np.issubdtype(actions.dtype, np.integer)
            or bool(((actions < 0) | (actions >= len(TARGET_EXPOSURE_LEVELS))).any())
        ):
            raise ValueError("Puffer actions must be valid categorical exposure indices")
        infos: list[dict[str, float | int]] = []
        for index, action in enumerate(actions):
            environment = self.envs[index]
            residual = int(action) / (len(TARGET_EXPOSURE_LEVELS) - 1) - 0.5
            observation, reward, terminated, truncated, info = self.envs[index].step_target(
                float(np.clip(environment.baseline_target_exposure() + residual, 0.0, 1.0))
            )
            episode_end = terminated or truncated
            self.rewards[index] = reward
            self.terminals[index] = episode_end
            self.truncations[index] = False
            self._episode_returns[index] += reward
            if episode_end:
                infos.append(
                    {
                        **info,
                        "episode_return": float(self._episode_returns[index]),
                        "episode_length": self.envs[index].index,
                    }
                )
                self._episodes[index] += 1
                self._reset_agent(index)
            else:
                self.observations[index] = observation
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self) -> None:
        for environment in self.envs:
            environment.close()
