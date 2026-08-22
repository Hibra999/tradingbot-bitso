from __future__ import annotations

import math
import random
import signal
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import AppConfig
from data import make_sliding_folds
from quant import (
    ALPHA_FORECAST_COLUMNS,
    ALPHA_TARGET_COLUMN,
    CausalAlphaEnsemble,
    CausalFeaturePipeline,
    atr,
    forward_return_targets,
)
from validation import (
    CPCVSplitter,
    PerturbationConfig,
    SeedHarness,
    combinatorial_pbo,
    institutional_metrics,
    model_confidence_set,
    moving_block_monte_carlo,
    paired_block_bootstrap_test,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

from .candidates import (
    PUFFER_ALGORITHM,
    PUFFER_AGENT_NAME,
    PufferPolicyRunner,
    build_puffer_policy,
    require_pufferlib,
)
from .environment import BracketTradingEnvV2, PufferTradingEnv
from .governance import dataframe_hash, dependency_versions, file_sha256, promotion_gate, write_manifest


def _gpu_postfix() -> dict[str, str]:
    try:
        import torch
    except ModuleNotFoundError:
        return {}
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    allocated, reserved = torch.cuda.memory_allocated() / gib, torch.cuda.memory_reserved() / gib
    return {"GPU memory": f"{allocated:.1f}/{reserved:.1f}GiB"}


def internal_purged_validation_tail(
    indices: np.ndarray,
    index: pd.DatetimeIndex,
    interval_end: pd.DatetimeIndex,
    *,
    validation_fraction: float = 0.10,
    embargo_bars: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(indices, dtype=int))
    if len(ordered) < 3 or not 0 < validation_fraction < 0.5:
        raise ValueError("not enough rows or invalid validation fraction")
    split = max(1, int(len(ordered) * (1 - validation_fraction)))
    validation = ordered[split:]
    training = ordered[:split]
    validation_start, validation_end = index[validation[0]], interval_end[validation].max()
    overlap = (index[training] <= validation_end) & (interval_end[training] >= validation_start)
    training = training[~overlap]
    if embargo_bars:
        cutoff = max(split - embargo_bars, 0)
        training = np.intersect1d(training, ordered[:cutoff], assume_unique=True)
    if not len(training) or not len(validation):
        raise ValueError("purging left an empty internal train/validation split")
    return training, validation


def _contiguous(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    ordered = np.sort(indices)
    return tuple(part for part in np.split(ordered, np.flatnonzero(np.diff(ordered) > 1) + 1) if len(part))


def _attach_scaled_features(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    aligned = raw.loc[features.index]
    return aligned.drop(columns=features.columns, errors="ignore").join(features)


@dataclass(frozen=True)
class CandidateDataset:
    symbol: str
    algorithm: str
    seed: int
    fold: int
    feature_columns: tuple[str, ...]
    training_segments: tuple[pd.DataFrame, ...]
    validation_segment: pd.DataFrame
    test_segments: tuple[pd.DataFrame, ...]
    m1_bars: pd.DataFrame
    output_dir: Path
    commission_rate: float = 0.001
    base_spread_bps: float = 2.0
    stress_spread_multiplier: float = 2.0
    stress_slippage_atr_fraction: float = 0.02
    deterministic_column: str = ALPHA_TARGET_COLUMN


@dataclass(frozen=True)
class _ResearchFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    episode_segments: tuple[np.ndarray, ...]
    test_groups: tuple[object, ...]


def _smoke_research_fold(
    index: pd.DatetimeIndex,
    interval_end: pd.DatetimeIndex,
    *,
    embargo_bars: int,
    max_holding_bars: int,
) -> _ResearchFold:
    if len(index) < 1_200:
        raise ValueError("smoke profile requires at least 1,200 complete H1 bars")
    smoke_embargo = min(embargo_bars, max_holding_bars)
    outer = CPCVSplitter(2, 1, smoke_embargo, max_holding_bars).split(index, interval_end)[-1]
    training, validation = internal_purged_validation_tail(
        outer.train_indices,
        index,
        interval_end,
        embargo_bars=smoke_embargo,
    )
    return _ResearchFold(training, validation, outer.episode_segments, tuple(outer.test_groups))


def _full_walk_forward_windows(
    development: pd.DataFrame,
    *,
    train_months: int,
    validation_months: int,
    evaluation_months: int,
    step_months: int,
    embargo_bars: int,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], int]:
    minimum_train_months = min(train_months, 12)
    if development.empty:
        raise ValueError("full validation requires non-empty development data")
    required_tail_months = validation_months + evaluation_months + step_months
    start, end = development.index.min(), development.index.max()
    effective_train_months = next(
        (
            months
            for months in range(train_months, minimum_train_months - 1, -1)
            if start + pd.DateOffset(months=months + required_tail_months) <= end
        ),
        None,
    )
    if effective_train_months is not None:
        windows = make_sliding_folds(
            development,
            train_years=effective_train_months / 12,
            val_months=validation_months,
            test_months=evaluation_months,
            step_months=step_months,
            embargo_bars=embargo_bars,
        )
        if len(windows) >= 2:
            return windows, effective_train_months
    raise ValueError(
        "full validation requires data for at least two complete walk-forward folds "
        f"with a minimum {minimum_train_months}-month training window"
    )


@dataclass(frozen=True)
class CandidateRun:
    returns: tuple[float, ...]
    artifact_path: str
    validation_score: float
    timestamps: tuple[str, ...] = ()
    training_returns: tuple[float, ...] = ()
    training_timestamps: tuple[str, ...] = ()
    benchmark_returns: tuple[float, ...] = ()
    training_benchmark_returns: tuple[float, ...] = ()
    validation_trials: tuple[float, ...] = ()
    stress_returns: tuple[float, ...] = ()
    stress_timestamps: tuple[str, ...] = ()
    deterministic_returns: tuple[float, ...] = ()
    training_deterministic_returns: tuple[float, ...] = ()


def _segment_return_index(segments: tuple[pd.DataFrame, ...]) -> pd.DatetimeIndex:
    values = [segment.index[1:] for segment in segments if len(segment) > 1]
    if not values:
        return pd.DatetimeIndex([])
    return values[0].append(values[1:])


def _run_series(
    result: CandidateRun,
    segments: tuple[pd.DataFrame, ...],
    *,
    training: bool = False,
) -> pd.Series:
    returns = result.training_returns if training else result.returns
    timestamps = result.training_timestamps if training else result.timestamps
    index = pd.DatetimeIndex(timestamps) if timestamps else _segment_return_index(segments)
    if len(index) != len(returns):
        raise ValueError("candidate returns and timestamps must align")
    series = pd.Series(returns, index=index, dtype=float).sort_index()
    if series.index.has_duplicates:
        raise ValueError("candidate return timestamps must be unique")
    return series


def _benchmark_series(
    result: CandidateRun,
    strategy: pd.Series,
    segments: tuple[pd.DataFrame, ...],
    *,
    training: bool = False,
) -> pd.Series:
    returns = result.training_benchmark_returns if training else result.benchmark_returns
    if returns:
        if len(returns) != len(strategy):
            raise ValueError("candidate benchmark and strategy returns must align")
        return pd.Series(returns, index=strategy.index, dtype=float)
    values = [segment["Close"].pct_change().dropna().astype(float) for segment in segments]
    benchmark = pd.concat(values).sort_index() if values else pd.Series(dtype=float)
    return benchmark.reindex(strategy.index)


def _stress_run_series(result: CandidateRun, segments: tuple[pd.DataFrame, ...]) -> pd.Series:
    if not result.stress_returns:
        return _run_series(result, segments)
    index = (
        pd.DatetimeIndex(result.stress_timestamps)
        if result.stress_timestamps
        else _segment_return_index(segments)
    )
    if len(index) != len(result.stress_returns):
        raise ValueError("stress returns and timestamps must align")
    return pd.Series(result.stress_returns, index=index, dtype=float).sort_index()


def _deterministic_series(
    result: CandidateRun,
    strategy: pd.Series,
    segments: tuple[pd.DataFrame, ...],
    *,
    training: bool = False,
) -> pd.Series:
    returns = result.training_deterministic_returns if training else result.deterministic_returns
    if returns:
        if len(returns) != len(strategy):
            raise ValueError("deterministic baseline and strategy returns must align")
        return pd.Series(returns, index=strategy.index, dtype=float)
    return pd.Series(np.zeros(len(strategy)), index=strategy.index, dtype=float)


def _certainty_equivalent(strategy: pd.Series, baseline: pd.Series) -> float:
    aligned = pd.concat((strategy.rename("strategy"), baseline.rename("baseline")), axis=1).dropna()
    if len(aligned) < 2:
        return float("-inf")
    active = aligned["strategy"] - aligned["baseline"]
    return float((active.mean() - 0.5 * active.var(ddof=1)) * 365 * 24)


def _volatility_matched_buy_and_hold(
    segments: tuple[pd.DataFrame, ...], round_trip_cost: float
) -> pd.Series:
    values: list[pd.Series] = []
    for segment in segments:
        returns = segment["Close"].pct_change().astype(float)
        annualized_volatility = returns.rolling(24).std(ddof=0) * np.sqrt(365 * 24)
        exposure = (0.20 / annualized_volatility.replace(0, np.nan)).clip(0, 1).fillna(0)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        result = exposure.shift() * returns - turnover.shift() * round_trip_cost / 2
        if len(result):
            result.iloc[-1] -= exposure.shift().iloc[-1] * round_trip_cost / 2
        values.append(result.iloc[1:])
    return pd.concat(values).sort_index() if values else pd.Series(dtype=float)


def _time_series_momentum(
    segments: tuple[pd.DataFrame, ...], horizon: int = 24, round_trip_cost: float = 0.0
) -> pd.Series:
    values: list[pd.Series] = []
    for segment in segments:
        returns = segment["Close"].pct_change().astype(float)
        exposure = (segment["Close"].pct_change(horizon) > 0).astype(float)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        result = exposure.shift() * returns - turnover.shift() * round_trip_cost / 2
        if len(result):
            result.iloc[-1] -= exposure.shift().iloc[-1] * round_trip_cost / 2
        values.append(result.iloc[1:])
    return pd.concat(values).sort_index() if values else pd.Series(dtype=float)


class PufferCandidateRunner:
    """External-machine runner; validation selects a checkpoint, tests only score it."""

    def __init__(
        self,
        timesteps: int = 100_000,
        evaluations: int = 5,
        notifier: object | None = None,
        parallel_envs: int = 16,
        bptt_horizon: int = 256,
        minibatch_size: int = 1_024,
    ):
        if min(timesteps, evaluations, parallel_envs, bptt_horizon, minibatch_size) < 1:
            raise ValueError("PuffeRL training values must be positive")
        if minibatch_size < bptt_horizon:
            raise ValueError("PuffeRL minibatches must cover one BPTT sequence")
        self.timesteps = timesteps
        self.evaluations = evaluations
        self.notifier = notifier
        self.parallel_envs = parallel_envs
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size

    @staticmethod
    def _environment(
        dataset: CandidateDataset,
        frame: pd.DataFrame,
        random_seed: int = 0,
        *,
        randomize: bool = True,
        stress: bool = False,
    ) -> BracketTradingEnvV2:
        return BracketTradingEnvV2(
            frame,
            dataset.m1_bars,
            list(dataset.feature_columns),
            action_mode="ppo",
            randomize=randomize,
            random_seed=random_seed,
            allow_short=False,
            commission_rate=dataset.commission_rate * (
                dataset.stress_spread_multiplier if stress else 1.0
            ),
            base_spread_bps=dataset.base_spread_bps * (
                dataset.stress_spread_multiplier if stress else 1.0
            ),
            perturbation_config=PerturbationConfig(
                slippage_atr_fraction=dataset.stress_slippage_atr_fraction
            ),
        )

    def _training_environment(
        self,
        dataset: CandidateDataset,
        segments: tuple[pd.DataFrame, ...],
        environment_count: int,
        episode_steps: int,
    ) -> PufferTradingEnv:
        return PufferTradingEnv(
            segments,
            dataset.m1_bars,
            list(dataset.feature_columns),
            num_agents=environment_count,
            episode_steps=episode_steps,
            random_seed=dataset.seed,
            commission_rate=dataset.commission_rate,
            base_spread_bps=dataset.base_spread_bps,
            perturbation_config=PerturbationConfig(
                slippage_atr_fraction=dataset.stress_slippage_atr_fraction
            ),
        )

    @staticmethod
    def _buy_and_hold(
        segments: tuple[pd.DataFrame, ...], round_trip_cost: float
    ) -> pd.Series:
        values = []
        for segment in segments:
            returns = segment["Close"].pct_change().dropna().astype(float)
            if len(returns):
                returns.iloc[0] -= round_trip_cost / 2
                returns.iloc[-1] -= round_trip_cost / 2
            values.append(returns)
        return pd.concat(values).sort_index() if values else pd.Series(dtype=float)

    def _evaluate(
        self,
        model,
        dataset: CandidateDataset,
        segments: tuple[pd.DataFrame, ...],
        *,
        randomize: bool = False,
        stress: bool = False,
        seed_offset: int = 0,
    ) -> pd.Series:
        returns: list[pd.Series] = []
        for number, segment in enumerate(segments, 1):
            environment = self._environment(
                dataset,
                segment,
                dataset.seed + seed_offset + number,
                randomize=randomize,
                stress=stress,
            )
            observation, _ = environment.reset(seed=dataset.seed + seed_offset + number)
            policy = PufferPolicyRunner(model)
            values: list[float] = []
            timestamps: list[pd.Timestamp] = []
            done, previous_equity = False, environment.core.equity
            started, last = time.monotonic(), 0.0
            status = (
                f"{dataset.symbol} | {PUFFER_AGENT_NAME} | fold {dataset.fold + 1} | "
                f"seed {dataset.seed} | evaluation segment {number}/{len(segments)}"
            )
            with tqdm(
                total=max(len(segment) - 1, 0),
                desc=f"Evaluation {number}/{len(segments)}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            ) as progress:
                while not done:
                    action = policy.predict(observation)
                    observation, _, terminated, truncated, info = environment.step(action)
                    done = terminated or truncated
                    equity = float(info["equity"])
                    values.append(equity / previous_equity - 1)
                    timestamps.append(segment.index[environment.index])
                    previous_equity = equity
                    progress.update()
                    now = time.monotonic()
                    if now - last >= 1.0:
                        publish = getattr(self.notifier, "progress", None)
                        if publish:
                            publish(status, int(progress.n), int(progress.total), started)
                        last = now
                publish = getattr(self.notifier, "progress", None)
                if publish and progress.total:
                    publish(status, int(progress.n), int(progress.total), started)
            returns.append(pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float))
        return pd.concat(returns).sort_index() if returns else pd.Series(dtype=float)

    def _evaluate_deterministic(
        self,
        dataset: CandidateDataset,
        segments: tuple[pd.DataFrame, ...],
        *,
        stress: bool = False,
        seed_offset: int = 0,
    ) -> pd.Series:
        returns: list[pd.Series] = []
        for number, segment in enumerate(segments, 1):
            environment = self._environment(
                dataset,
                segment,
                dataset.seed + seed_offset + number,
                randomize=stress,
                stress=stress,
            )
            environment.reset(seed=dataset.seed + seed_offset + number)
            values: list[float] = []
            timestamps: list[pd.Timestamp] = []
            previous_equity = environment.core.equity
            for target in segment[dataset.deterministic_column].iloc[:-1].to_numpy(dtype=float):
                _, _, _, _, info = environment.step_target(float(target))
                equity = float(info["equity"])
                values.append(equity / previous_equity - 1)
                timestamps.append(segment.index[environment.index])
                previous_equity = equity
            returns.append(pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float))
        return pd.concat(returns).sort_index() if returns else pd.Series(dtype=float)

    def __call__(self, dataset: CandidateDataset) -> CandidateRun:
        require_pufferlib()
        import torch

        warning_filters = warnings.filters.copy()
        sigint_handler = signal.getsignal(signal.SIGINT)
        try:
            from pufferlib.pufferl import PuffeRL
        finally:
            warnings.filters[:] = warning_filters
            signal.signal(signal.SIGINT, sigint_handler)

        segments = dataset.training_segments
        if not segments:
            raise ValueError("training segments must not be empty")
        if dataset.algorithm != PUFFER_ALGORITHM:
            raise ValueError("PufferLib 3.0 is the only supported RL algorithm")

        environment_count = max(len(segments), self.parallel_envs)
        rollout_size = environment_count * self.bptt_horizon
        rollout_count = math.ceil(self.timesteps / rollout_size)
        if rollout_count < self.evaluations:
            raise ValueError("PuffeRL timesteps must cover at least one rollout per evaluation")
        chunk, remainder = divmod(rollout_count, self.evaluations)
        rollouts_by_evaluation = tuple(
            chunk + (evaluation < remainder) for evaluation in range(self.evaluations)
        )
        total_steps = rollout_count * rollout_size
        minibatch_segments = min(environment_count, self.minibatch_size // self.bptt_horizon)
        while environment_count % minibatch_segments:
            minibatch_segments -= 1
        minibatch_size = minibatch_segments * self.bptt_horizon

        random.seed(dataset.seed)
        np.random.seed(dataset.seed)
        torch.manual_seed(dataset.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(dataset.seed)

        validation_baseline = self._evaluate_deterministic(dataset, (dataset.validation_segment,))
        training_env = self._training_environment(
            dataset,
            segments,
            environment_count,
            rollout_count * self.bptt_horizon,
        )
        model = build_puffer_policy(training_env)
        device = str(next(model.parameters()).device)
        trainer = PuffeRL(
            {
                "env": "tradingbot_bitso",
                "seed": dataset.seed,
                "torch_deterministic": True,
                "device": device,
                "cpu_offload": False,
                "optimizer": "adam",
                "precision": "float32",
                "total_timesteps": total_steps + rollout_size,
                "learning_rate": 3e-4,
                "anneal_lr": False,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "update_epochs": 10,
                "clip_coef": 0.2,
                "vf_coef": 0.5,
                "vf_clip_coef": 0.2,
                "max_grad_norm": 0.5,
                "ent_coef": 0.001,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "data_dir": str(dataset.output_dir),
                "checkpoint_interval": rollout_count + 1,
                "batch_size": rollout_size,
                "minibatch_size": minibatch_size,
                "max_minibatch_size": minibatch_size,
                "bptt_horizon": self.bptt_horizon,
                "compile": False,
                "compile_fullgraph": False,
                "compile_mode": "default",
                "use_rnn": True,
                "vtrace_rho_clip": 1.0,
                "vtrace_c_clip": 1.0,
                "prio_alpha": 0.8,
                "prio_beta0": 0.2,
            },
            training_env,
            model,
        )
        trainer.print_dashboard = lambda *args, **kwargs: None  # tqdm owns terminal rendering.
        if device == "cuda":
            torch.backends.cudnn.benchmark = False
        best_score = float("-inf")
        validation_scores: list[float] = []
        artifact = dataset.output_dir / "best_model.pt"
        label = (
            f"{dataset.symbol} | {PUFFER_AGENT_NAME} | fold {dataset.fold + 1} | "
            f"seed {dataset.seed} | {environment_count} envs"
        )
        started = time.monotonic()
        try:
            with tqdm(
                total=total_steps,
                desc=f"{dataset.symbol} {PUFFER_AGENT_NAME} f{dataset.fold + 1} s{dataset.seed}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            ) as progress:
                for evaluation, rollouts in enumerate(rollouts_by_evaluation, 1):
                    status = f"{label} | evaluation {evaluation}/{self.evaluations}"
                    for _ in range(rollouts):
                        before = trainer.global_step
                        trainer.evaluate()
                        trainer.train()
                        progress.update(trainer.global_step - before)
                        publish = getattr(self.notifier, "progress", None)
                        if publish:
                            publish(status, int(progress.n), int(progress.total), started)
                    update = getattr(self.notifier, "update", None)
                    if update:
                        update(f"{status} | validation")
                    validation_returns = self._evaluate(
                        model, dataset, (dataset.validation_segment,)
                    )
                    score = _certainty_equivalent(validation_returns, validation_baseline)
                    validation_scores.append(
                        sharpe_ratio(validation_returns.dropna().to_numpy())
                    )
                    if evaluation == 1 or np.isfinite(score) and (
                        not np.isfinite(best_score) or score > best_score
                    ):
                        best_score = score
                        torch.save(model.state_dict(), artifact)
                    model.train()
                    progress.set_postfix(
                        envs=environment_count,
                        evaluation=f"{evaluation}/{self.evaluations}",
                        best=f"{best_score:.4f}",
                        **_gpu_postfix(),
                    )
        finally:
            trainer.utilization.stop()
            trainer.utilization.join(timeout=2.0)
            training_env.close()

        model.load_state_dict(torch.load(artifact, map_location=device, weights_only=True))
        model.eval()
        del trainer
        if device == "cuda":
            torch.cuda.empty_cache()
        update = getattr(self.notifier, "update", None)
        if update:
            update(f"{label} | test evaluation")
        training_returns = self._evaluate(model, dataset, dataset.training_segments)
        test_returns = self._evaluate(model, dataset, dataset.test_segments)
        training_deterministic = self._evaluate_deterministic(dataset, dataset.training_segments)
        test_deterministic = self._evaluate_deterministic(dataset, dataset.test_segments)
        stress_returns = self._evaluate(
            model,
            dataset,
            dataset.test_segments,
            randomize=True,
            stress=True,
            seed_offset=10_000,
        )
        return CandidateRun(
            tuple(test_returns.to_numpy()),
            str(artifact),
            best_score,
            tuple(str(value) for value in test_returns.index),
            tuple(training_returns.to_numpy()),
            tuple(str(value) for value in training_returns.index),
            tuple(
                self._buy_and_hold(
                    dataset.test_segments,
                    2 * dataset.commission_rate + dataset.base_spread_bps / 10_000,
                ).reindex(test_returns.index).to_numpy()
            ),
            tuple(
                self._buy_and_hold(
                    dataset.training_segments,
                    2 * dataset.commission_rate + dataset.base_spread_bps / 10_000,
                ).reindex(training_returns.index).to_numpy()
            ),
            tuple(validation_scores),
            tuple(stress_returns.to_numpy()),
            tuple(str(value) for value in stress_returns.index),
            tuple(test_deterministic.reindex(test_returns.index).to_numpy()),
            tuple(training_deterministic.reindex(training_returns.index).to_numpy()),
        )


class TrainingEngine:
    def __init__(
        self,
        config: AppConfig,
        runner: Callable[[CandidateDataset], CandidateRun] | None = None,
        notifier: object | None = None,
    ):
        self.config = config
        self.runner = runner or PufferCandidateRunner(
            config.rl.timesteps,
            config.rl.evaluations,
            notifier,
            config.rl.puffer_envs,
            config.rl.bptt_horizon,
            config.rl.minibatch_size,
        )
        self.notifier = notifier

    @staticmethod
    def _git_sha() -> str:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def run_symbol(self, symbol: str, decision_data: pd.DataFrame, m1_data: pd.DataFrame) -> dict[str, object]:
        if not self.config.rl.algorithms:
            raise ValueError("at least one RL algorithm must be enabled")
        decision_data = decision_data.copy()
        if "atr" not in decision_data:
            decision_data["atr"] = atr(decision_data)
        holding = self.config.validation.max_holding_bars
        end_positions = np.minimum(np.arange(len(decision_data)) + holding, len(decision_data) - 1)
        interval_end = pd.DatetimeIndex(decision_data.index[end_positions])
        alpha_targets = forward_return_targets(decision_data.index, m1_data)
        round_trip_cost = (
            2 * self.config.validation.commission_bps + self.config.validation.base_spread_bps
        ) / 10_000
        holdout_start: pd.Timestamp | None = None
        holdout_end: pd.Timestamp | None = None
        effective_train_months: int | None = None
        folds: list[_ResearchFold] = []
        if self.config.profile == "full":
            latest = decision_data.index.max()
            holdout_end = pd.Timestamp(latest.year, latest.month, 1, tz=latest.tz)
            holdout_start = holdout_end - pd.DateOffset(months=self.config.validation.holdout_months)
            development = decision_data.loc[decision_data.index < holdout_start]
            windows, effective_train_months = _full_walk_forward_windows(
                development,
                train_months=self.config.validation.train_months,
                validation_months=self.config.validation.validation_months,
                evaluation_months=self.config.validation.evaluation_months,
                step_months=self.config.validation.step_months,
                embargo_bars=self.config.validation.embargo_bars,
            )
            if effective_train_months < self.config.validation.train_months:
                update = getattr(self.notifier, "update", None)
                if update:
                    update(
                        f"{symbol} | walk-forward training window adapted | "
                        f"{self.config.validation.train_months}m -> {effective_train_months}m"
                    )
            for training, validation, evaluation in windows:
                train_indices = decision_data.index.get_indexer(training.index)
                validation_indices = decision_data.index.get_indexer(validation.index)
                evaluation_indices = decision_data.index.get_indexer(evaluation.index)
                if min(train_indices.min(), validation_indices.min(), evaluation_indices.min()) < 0:
                    raise ValueError("walk-forward timestamps must map to the source data")
                folds.append(
                    _ResearchFold(
                        train_indices,
                        validation_indices,
                        (evaluation_indices,),
                        (str(evaluation.index.min()), str(evaluation.index.max())),
                    )
                )
        else:
            folds.append(
                _smoke_research_fold(
                    decision_data.index,
                    interval_end,
                    embargo_bars=self.config.validation.embargo_bars,
                    max_holding_bars=holding,
                )
            )
        seeds = self.config.validation.full_seeds if self.config.profile == "full" else self.config.validation.smoke_seeds
        algorithms: dict[str, object] = {}
        all_artifacts: list[str] = []
        eligible_artifacts: list[str] = []
        all_prior = np.arange(len(decision_data))
        prepared = []
        alpha_trial_ledger: list[dict[str, object]] = []
        preparation_started = time.monotonic()
        for fold_number, fold in tqdm(
            enumerate(folds),
            total=len(folds),
            desc=f"{symbol} feature preparation",
            leave=False,
            dynamic_ncols=True,
        ):
            update = getattr(self.notifier, "update", None)
            if update:
                update(f"{symbol} | feature preparation | fold {fold_number + 1}/{len(folds)}")
            train_indices, validation_indices = fold.train_indices, fold.validation_indices
            pipeline = CausalFeaturePipeline(
                config_hash=self.config.config_hash,
                fracdiff_threshold=1e-5 if self.config.profile == "full" else 1e-3,
                random_state=fold_number,
            )
            pipeline.fit(decision_data.iloc[train_indices])

            def transformed(indices: np.ndarray, history_indices: np.ndarray) -> pd.DataFrame:
                raw = decision_data.iloc[indices]
                prior = history_indices[history_indices < indices.min()]
                features = pipeline.transform(raw, history_context=decision_data.iloc[prior])
                return _attach_scaled_features(raw, features)

            raw_training_segments = tuple(
                transformed(segment, train_indices) for segment in _contiguous(train_indices)
            )
            alpha = CausalAlphaEnsemble(random_state=fold_number).fit(
                pd.concat(
                    [segment.loc[:, pipeline.feature_order] for segment in raw_training_segments]
                ).sort_index(),
                alpha_targets,
            )

            def with_alpha(frame: pd.DataFrame) -> pd.DataFrame:
                forecasts = alpha.transform(
                    frame.loc[:, pipeline.feature_order], round_trip_cost=round_trip_cost
                )
                return frame.join(forecasts).join(alpha_targets.reindex(frame.index))

            alpha_path = (
                self.config.models_dir
                / symbol.replace("/", "_")
                / "features"
                / f"fold_{fold_number}_alpha.pkl"
            )
            alpha.save(alpha_path)
            feature_order = pipeline.feature_order + ALPHA_FORECAST_COLUMNS
            feature_manifest = pipeline.manifest()
            feature_manifest["feature_order"] = list(feature_order)
            feature_manifest["alpha"] = alpha.manifest()
            for horizon, experts in alpha.manifest()["experts"].items():
                for expert in experts:
                    alpha_trial_ledger.append(
                        {
                            "kind": "alpha_expert",
                            "fold": fold_number,
                            "horizon_hours": int(horizon),
                            **expert,
                        }
                    )

            prepared.append(
                (
                    fold_number,
                    fold,
                    feature_order,
                    tuple(with_alpha(segment) for segment in raw_training_segments),
                    with_alpha(transformed(validation_indices, train_indices)),
                    tuple(
                        with_alpha(transformed(segment, all_prior))
                        for segment in fold.episode_segments
                    ),
                    feature_manifest,
                    str(alpha_path.resolve()),
                )
            )
            progress = getattr(self.notifier, "progress", None)
            if progress:
                progress(
                    f"{symbol} | feature preparation | fold {fold_number + 1}/{len(folds)}",
                    fold_number + 1,
                    len(folds),
                    preparation_started,
                )

        candidates_by_algorithm: dict[str, list[dict[str, object]]] = {}
        trial_sharpes: list[float] = []
        trial_ledger: list[dict[str, object]] = list(alpha_trial_ledger)
        for algorithm in tqdm(
            self.config.rl.algorithms,
            desc=f"{symbol} {PUFFER_AGENT_NAME}",
            leave=False,
            dynamic_ncols=True,
        ):
            candidates: list[dict[str, object]] = []
            jobs = tqdm(
                total=len(prepared) * len(seeds),
                desc=f"{PUFFER_AGENT_NAME} fold/seed jobs",
                leave=False,
                dynamic_ncols=True,
            )
            for (
                fold_number,
                fold,
                feature_order,
                training_segments,
                validation_segment,
                test_segments,
                feature_manifest,
                alpha_artifact_path,
            ) in prepared:
                for seed in seeds:
                    jobs.set_postfix(fold=f"{fold_number + 1}/{len(folds)}", seed=seed)
                    output_dir = self.config.models_dir / symbol.replace("/", "_") / algorithm / f"fold_{fold_number}" / f"seed_{seed}"
                    dataset = CandidateDataset(
                        symbol,
                        algorithm,
                        seed,
                        fold_number,
                        feature_order,
                        training_segments,
                        validation_segment,
                        test_segments,
                        m1_data,
                        output_dir,
                        commission_rate=self.config.validation.commission_bps / 10_000,
                        base_spread_bps=self.config.validation.base_spread_bps,
                        stress_spread_multiplier=self.config.validation.stress_spread_multiplier,
                        stress_slippage_atr_fraction=self.config.validation.stress_slippage_atr_fraction,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result = self.runner(dataset)
                    evaluation = _run_series(result, test_segments)
                    stress = _stress_run_series(result, test_segments)
                    training = _run_series(result, training_segments, training=True)
                    evaluation_benchmark = _benchmark_series(result, evaluation, test_segments)
                    training_benchmark = _benchmark_series(result, training, training_segments, training=True)
                    evaluation_deterministic = _deterministic_series(result, evaluation, test_segments)
                    training_deterministic = _deterministic_series(
                        result, training, training_segments, training=True
                    )
                    evaluation_volatility_benchmark = _volatility_matched_buy_and_hold(
                        test_segments, round_trip_cost
                    ).reindex(evaluation.index)
                    evaluation_momentum = {
                        str(horizon): _time_series_momentum(
                            test_segments, horizon, round_trip_cost
                        ).reindex(evaluation.index)
                        for horizon in (12, 24, 168)
                    }
                    test_sharpe = sharpe_ratio(evaluation.to_numpy()) if len(evaluation) > 1 else float("nan")
                    alpha_pairs = pd.concat(
                        [
                            segment[["alpha_mean_12h", "target_12h"]]
                            for segment in test_segments
                        ]
                    ).dropna()
                    alpha_ic = float(
                        alpha_pairs["alpha_mean_12h"].corr(
                            alpha_pairs["target_12h"], method="spearman"
                        )
                    )
                    artifact_path = str(Path(result.artifact_path).resolve())
                    candidate = {
                        "fold": fold_number,
                        "seed": seed,
                        "test_groups": list(fold.test_groups),
                        "validation_score": result.validation_score,
                        "test_sharpe": test_sharpe,
                        "alpha_ic_12h": alpha_ic,
                        "stress_return": float(np.prod(1 + stress.to_numpy()) - 1),
                        "artifact_path": artifact_path,
                        "feature_manifest": feature_manifest,
                        "alpha_artifact_path": alpha_artifact_path,
                        "evaluation": evaluation,
                        "stress": stress,
                        "training": training,
                        "evaluation_benchmark": evaluation_benchmark,
                        "training_benchmark": training_benchmark,
                        "evaluation_deterministic": evaluation_deterministic,
                        "training_deterministic": training_deterministic,
                        "evaluation_volatility_benchmark": evaluation_volatility_benchmark,
                        "evaluation_momentum": evaluation_momentum,
                    }
                    candidates.append(candidate)
                    all_artifacts.append(artifact_path)
                    trial_count = max(1, len(result.validation_trials))
                    finite_trials = [
                        float(value) for value in result.validation_trials if np.isfinite(value)
                    ]
                    if finite_trials:
                        trial_sharpes.extend(finite_trials)
                    elif np.isfinite(result.validation_score):
                        trial_sharpes.append(float(result.validation_score))
                    trial_ledger.append(
                        {
                            "algorithm": algorithm,
                            "fold": fold_number,
                            "seed": seed,
                            "checkpoint_trials": trial_count,
                            "validation_scores": list(result.validation_trials) or [result.validation_score],
                            "evaluation_sharpe": test_sharpe,
                            "stress_return": candidate["stress_return"],
                            "artifact_path": artifact_path,
                        }
                    )
                    update = getattr(self.notifier, "update", None)
                    if update:
                        update(
                            f"{symbol} | {PUFFER_AGENT_NAME} | fold {fold_number + 1}/{len(folds)} | "
                            f"seed {seed} complete"
                        )
                    jobs.update()
            jobs.close()
            candidates_by_algorithm[algorithm] = candidates

        configurations = [(algorithm, seed) for algorithm in self.config.rl.algorithms for seed in seeds]
        selection_matrix = np.asarray(
            [
                [
                    next(
                        float(candidate["validation_score"])
                        for candidate in candidates_by_algorithm[algorithm]
                        if candidate["fold"] == fold_number and candidate["seed"] == seed
                    )
                    for algorithm, seed in configurations
                ]
                for fold_number in range(len(folds))
            ]
        )
        evaluation_matrix = np.asarray(
            [
                [
                    next(
                        float(candidate["test_sharpe"])
                        for candidate in candidates_by_algorithm[algorithm]
                        if candidate["fold"] == fold_number and candidate["seed"] == seed
                    )
                    for algorithm, seed in configurations
                ]
                for fold_number in range(len(folds))
            ]
        )
        if self.config.profile == "full" and len(configurations) >= 2:
            configuration_paths = []
            for algorithm, seed in configurations:
                path = pd.concat(
                    [
                        candidate["evaluation"]
                        for candidate in sorted(
                            candidates_by_algorithm[algorithm], key=lambda item: int(item["fold"])
                        )
                        if candidate["seed"] == seed
                    ]
                ).sort_index()
                if path.index.has_duplicates:
                    raise ValueError("CSCV candidate return paths must not overlap")
                configuration_paths.append(path.rename(f"{algorithm}:{seed}"))
            return_matrix = pd.concat(configuration_paths, axis=1, join="inner").dropna()
            pbo = combinatorial_pbo(return_matrix.to_numpy())
        else:
            pbo = (
                probability_of_backtest_overfitting(selection_matrix, evaluation_matrix)
                if len(configurations) >= 2 and len(folds) >= 2
                else {"pbo_probability": 1.0, "pbo_median_logit": float("nan"), "pbo_folds": 0.0}
            )

        report_candidates: list[dict[str, object]] = []
        for algorithm, candidates in candidates_by_algorithm.items():
            seed_candidates = {
                seed: sorted(
                    (item for item in candidates if item["seed"] == seed),
                    key=lambda item: int(item["fold"]),
                )
                for seed in seeds
            }
            seed_paths: dict[int, pd.Series] = {}
            seed_stress_paths: dict[int, pd.Series] = {}
            seed_deterministic_paths: dict[int, pd.Series] = {}
            seed_validation_scores: dict[int, float] = {}
            for seed, items in seed_candidates.items():
                if self.config.profile == "full":
                    path = pd.concat([item["evaluation"] for item in items]).sort_index()
                    stress_path = pd.concat([item["stress"] for item in items]).sort_index()
                    deterministic_path = pd.concat(
                        [item["evaluation_deterministic"] for item in items]
                    ).sort_index()
                    if path.index.has_duplicates:
                        raise ValueError("walk-forward evaluation timestamps must not overlap")
                    if stress_path.index.has_duplicates:
                        raise ValueError("walk-forward stress timestamps must not overlap")
                    validation_score = float(
                        np.mean([float(item["validation_score"]) for item in items])
                    )
                else:
                    representative = max(items, key=lambda item: float(item["validation_score"]))
                    path = representative["evaluation"]
                    stress_path = representative["stress"]
                    deterministic_path = representative["evaluation_deterministic"]
                    validation_score = float(representative["validation_score"])
                seed_paths[seed] = path
                seed_stress_paths[seed] = stress_path
                seed_deterministic_paths[seed] = deterministic_path
                seed_validation_scores[seed] = validation_score
            selected_seed = max(seed_validation_scores, key=seed_validation_scores.get)
            selected = (
                seed_candidates[selected_seed][-1]
                if self.config.profile == "full"
                else max(
                    seed_candidates[selected_seed],
                    key=lambda item: float(item["validation_score"]),
                )
            )
            evaluation = seed_paths[selected_seed]
            evaluation_benchmark = (
                pd.concat(
                    [item["evaluation_benchmark"] for item in seed_candidates[selected_seed]]
                ).sort_index()
                if self.config.profile == "full"
                else selected["evaluation_benchmark"]
            )
            evaluation_deterministic = seed_deterministic_paths[selected_seed]
            evaluation_volatility_benchmark = (
                pd.concat(
                    [
                        item["evaluation_volatility_benchmark"]
                        for item in seed_candidates[selected_seed]
                    ]
                ).sort_index()
                if self.config.profile == "full"
                else selected["evaluation_volatility_benchmark"]
            )
            evaluation_momentum = {
                str(horizon): (
                    pd.concat(
                        [
                            item["evaluation_momentum"][str(horizon)]
                            for item in seed_candidates[selected_seed]
                        ]
                    ).sort_index()
                    if self.config.profile == "full"
                    else selected["evaluation_momentum"][str(horizon)]
                )
                for horizon in (12, 24, 168)
            }
            metrics = institutional_metrics(
                evaluation.to_numpy(),
                trial_sharpes=trial_sharpes,
                bootstrap_repetitions=2_000 if self.config.profile == "full" else 100,
            )
            seed_evaluation = SeedHarness(seeds, smoke=self.config.profile == "smoke").run(
                lambda seed: {
                    "sharpe": sharpe_ratio(seed_paths[seed].to_numpy()),
                    "dsr_z": metrics["dsr_z"],
                    "return": float(np.prod(1 + seed_paths[seed].to_numpy()) - 1),
                }
            )
            monte_carlo = moving_block_monte_carlo(
                evaluation.to_numpy(),
                paths=(
                    self.config.validation.monte_carlo_paths
                    if self.config.profile == "full"
                    else min(100, self.config.validation.monte_carlo_paths)
                ),
                block_size=min(24, len(evaluation)),
            )
            metrics["ruin_probability_20"] = monte_carlo.ruin_probability_20
            metrics["ruin_probability_30"] = monte_carlo.ruin_probability_30
            strategy_return = float(np.prod(1 + evaluation.to_numpy()) - 1)
            buy_and_hold_return = float(np.prod(1 + evaluation_benchmark.to_numpy()) - 1)
            metrics["buy_and_hold_return"] = buy_and_hold_return
            metrics["excess_return_vs_buy_and_hold"] = strategy_return - buy_and_hold_return
            deterministic_return = float(
                np.prod(1 + evaluation_deterministic.to_numpy()) - 1
            )
            metrics["deterministic_alpha_return"] = deterministic_return
            metrics["excess_return_vs_deterministic_alpha"] = strategy_return - deterministic_return
            alpha_cash_test = paired_block_bootstrap_test(
                evaluation_deterministic.to_numpy(),
                np.zeros(len(evaluation_deterministic)),
                repetitions=2_000 if self.config.profile == "full" else 100,
                seed=4,
            )
            metrics["deterministic_alpha_ci95_low"] = alpha_cash_test["ci_low"]
            metrics["alpha_diagnostic_pass"] = all(
                bool(item["feature_manifest"]["alpha"]["diagnostic_pass"])
                for item in candidates
            )
            fold_alpha_ic = np.asarray(
                [float(item["alpha_ic_12h"]) for item in seed_candidates[selected_seed]]
            )
            metrics["alpha_ic_mean"] = float(np.nanmean(fold_alpha_ic))
            metrics["alpha_ic_positive_fraction"] = float(np.mean(fold_alpha_ic > 0))
            metrics["volatility_matched_buy_and_hold_return"] = float(
                np.prod(1 + evaluation_volatility_benchmark.dropna().to_numpy()) - 1
            )
            for horizon, momentum in evaluation_momentum.items():
                metrics[f"momentum_{horizon}h_return"] = float(
                    np.prod(1 + momentum.dropna().to_numpy()) - 1
                )
            alpha_test = paired_block_bootstrap_test(
                evaluation.to_numpy(),
                evaluation_deterministic.reindex(evaluation.index).to_numpy(),
                repetitions=2_000 if self.config.profile == "full" else 100,
            )
            volatility_test = paired_block_bootstrap_test(
                evaluation.to_numpy(),
                evaluation_volatility_benchmark.reindex(evaluation.index).to_numpy(),
                repetitions=2_000 if self.config.profile == "full" else 100,
                seed=1,
            )
            confidence_set = model_confidence_set(
                {
                    PUFFER_AGENT_NAME: evaluation.to_numpy(),
                    "Alpha": evaluation_deterministic.reindex(evaluation.index).to_numpy(),
                    "Vol B&H": evaluation_volatility_benchmark.reindex(evaluation.index).to_numpy(),
                },
                repetitions=2_000 if self.config.profile == "full" else 100,
                seed=2,
            )
            metrics["paired_alpha_ci95_low"] = alpha_test["ci_low"]
            metrics["paired_volatility_bh_ci95_low"] = volatility_test["ci_low"]
            metrics["mcs_90_pass"] = bool(
                confidence_set["retained"] == [PUFFER_AGENT_NAME]
                and set(confidence_set["eliminated"]) == {"Alpha", "Vol B&H"}
            )
            metrics["model_confidence_set"] = confidence_set
            metrics["stress_return"] = float(
                np.prod(1 + seed_stress_paths[selected_seed].to_numpy()) - 1
            )
            fold_returns = np.asarray(
                [
                    float(np.prod(1 + item["evaluation"].to_numpy()) - 1)
                    for item in seed_candidates[selected_seed]
                ]
            )
            positive_fold_returns = fold_returns[fold_returns > 0]
            metrics["profitable_fold_fraction"] = float(np.mean(fold_returns > 0))
            metrics["max_fold_profit_share"] = (
                float(positive_fold_returns.max() / positive_fold_returns.sum())
                if len(positive_fold_returns)
                else 1.0
            )
            metrics["seed_stability_pass"] = seed_evaluation.sri_pass
            metrics["seed_iqm_return_ci95_low"] = seed_evaluation.aggregate.get(
                "return", {}
            ).get("ci95_low", float("-inf"))
            metrics.update(pbo)
            algorithm_eligible, algorithm_reasons = promotion_gate(metrics, profile=self.config.profile)
            if algorithm_eligible and self.config.profile != "full":
                eligible_artifacts.append(str(selected["artifact_path"]))
            notify = getattr(self.notifier, "notify", None)
            if notify:
                notify(
                    f"{symbol} | {PUFFER_AGENT_NAME} | {'PASS' if algorithm_eligible else 'FAIL'} | "
                    f"Sharpe {metrics['sharpe']:.4f} | SQN {metrics['sqn']:.4f}"
                )
            report_candidates.append(
                {
                    **selected,
                    "eligible": algorithm_eligible,
                    "algorithm": algorithm,
                    "validation_score": seed_validation_scores[selected_seed],
                    "evaluation": evaluation,
                    "evaluation_benchmark": evaluation_benchmark,
                    "evaluation_deterministic": evaluation_deterministic,
                    "evaluation_volatility_benchmark": evaluation_volatility_benchmark,
                    "evaluation_momentum": evaluation_momentum,
                    "selected_seed": selected_seed,
                }
            )
            algorithms[algorithm] = {
                "eligible": algorithm_eligible if self.config.profile != "full" else False,
                "development_eligible": algorithm_eligible,
                "gate_reasons": algorithm_reasons,
                "selected_artifact": selected["artifact_path"] if self.config.profile != "full" else None,
                "selection_rule": "highest mean validation certainty-equivalent versus deterministic alpha",
                "selected_seed": selected_seed,
                "development_evaluation_range": [str(evaluation.index.min()), str(evaluation.index.max())],
                "fold_results": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in {
                            "evaluation",
                            "stress",
                            "training",
                            "evaluation_benchmark",
                            "training_benchmark",
                            "evaluation_deterministic",
                            "training_deterministic",
                            "evaluation_volatility_benchmark",
                            "evaluation_momentum",
                            "feature_manifest",
                            "alpha_artifact_path",
                        }
                    }
                    for candidate in candidates
                ],
                "seed_evaluation": seed_evaluation.manifest(),
                "metrics": metrics,
                "monte_carlo": monte_carlo.manifest(),
                "feature_manifest": selected["feature_manifest"],
            }

        development_champion = max(
            (item for item in report_candidates if item["eligible"]),
            key=lambda item: float(item["validation_score"]),
            default=max(report_candidates, key=lambda item: float(item["validation_score"])),
        )
        report_candidate = development_champion
        final_evaluation: dict[str, object] | None = None
        if self.config.profile == "full":
            champion_algorithm = str(development_champion["algorithm"])
            fallback = max(
                candidates_by_algorithm[champion_algorithm],
                key=lambda item: float(item["validation_score"]),
            )
            report_candidate = {**fallback, "algorithm": champion_algorithm, "eligible": False}
            if development_champion["eligible"]:
                if holdout_start is None or holdout_end is None:
                    raise RuntimeError("full validation must reserve a sealed holdout")
                validation_start = holdout_start - pd.DateOffset(
                    months=self.config.validation.validation_months
                )
                validation_start_position = int(decision_data.index.searchsorted(validation_start, side="left"))
                holdout_start_position = int(decision_data.index.searchsorted(holdout_start, side="left"))
                holdout_end_position = int(decision_data.index.searchsorted(holdout_end, side="left"))
                embargo = self.config.validation.embargo_bars
                final_train_indices = np.arange(max(validation_start_position - embargo, 0))
                final_validation_indices = np.arange(
                    min(validation_start_position + embargo, holdout_start_position),
                    holdout_start_position,
                )
                final_holdout_indices = np.arange(
                    min(holdout_start_position + embargo, holdout_end_position),
                    holdout_end_position,
                )
                if min(
                    len(final_train_indices),
                    len(final_validation_indices),
                    len(final_holdout_indices),
                ) < 2:
                    raise ValueError("sealed holdout split is too short after embargo")

                final_pipeline = CausalFeaturePipeline(
                    config_hash=self.config.config_hash,
                    random_state=int(development_champion["selected_seed"]),
                )
                final_pipeline.fit(decision_data.iloc[final_train_indices])

                def final_transformed(indices: np.ndarray, history_indices: np.ndarray) -> pd.DataFrame:
                    raw = decision_data.iloc[indices]
                    prior = history_indices[history_indices < indices.min()]
                    features = final_pipeline.transform(raw, history_context=decision_data.iloc[prior])
                    return _attach_scaled_features(raw, features)

                final_training_raw = final_transformed(final_train_indices, final_train_indices)
                final_alpha = CausalAlphaEnsemble(
                    random_state=int(development_champion["selected_seed"])
                ).fit(
                    final_training_raw.loc[:, final_pipeline.feature_order],
                    alpha_targets,
                )

                def with_final_alpha(frame: pd.DataFrame) -> pd.DataFrame:
                    return frame.join(
                        final_alpha.transform(
                            frame.loc[:, final_pipeline.feature_order],
                            round_trip_cost=round_trip_cost,
                        )
                    ).join(alpha_targets.reindex(frame.index))

                final_training = with_final_alpha(final_training_raw)
                final_validation = with_final_alpha(
                    final_transformed(final_validation_indices, final_train_indices)
                )
                final_holdout = with_final_alpha(
                    final_transformed(final_holdout_indices, all_prior)
                )
                feature_path = self.config.models_dir / symbol.replace("/", "_") / "final" / "features.pkl"
                alpha_path = self.config.models_dir / symbol.replace("/", "_") / "final" / "alpha.pkl"
                final_pipeline.save(feature_path)
                final_alpha.save(alpha_path)
                final_feature_order = final_pipeline.feature_order + ALPHA_FORECAST_COLUMNS
                final_feature_manifest = final_pipeline.manifest()
                final_feature_manifest["feature_order"] = list(final_feature_order)
                final_feature_manifest["alpha"] = final_alpha.manifest()
                final_candidates: list[dict[str, object]] = []
                for seed in seeds:
                    output_dir = (
                        self.config.models_dir
                        / symbol.replace("/", "_")
                        / champion_algorithm
                        / "final"
                        / f"seed_{seed}"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    dataset = CandidateDataset(
                        symbol,
                        champion_algorithm,
                        seed,
                        len(folds),
                        final_feature_order,
                        (final_training,),
                        final_validation,
                        (final_holdout,),
                        m1_data,
                        output_dir,
                        commission_rate=self.config.validation.commission_bps / 10_000,
                        base_spread_bps=self.config.validation.base_spread_bps,
                        stress_spread_multiplier=self.config.validation.stress_spread_multiplier,
                        stress_slippage_atr_fraction=self.config.validation.stress_slippage_atr_fraction,
                    )
                    result = self.runner(dataset)
                    training = _run_series(result, dataset.training_segments, training=True)
                    evaluation = _run_series(result, dataset.test_segments)
                    stress = _stress_run_series(result, dataset.test_segments)
                    training_benchmark = _benchmark_series(
                        result, training, dataset.training_segments, training=True
                    )
                    evaluation_benchmark = _benchmark_series(result, evaluation, dataset.test_segments)
                    training_deterministic = _deterministic_series(
                        result, training, dataset.training_segments, training=True
                    )
                    evaluation_deterministic = _deterministic_series(
                        result, evaluation, dataset.test_segments
                    )
                    evaluation_volatility_benchmark = _volatility_matched_buy_and_hold(
                        dataset.test_segments, round_trip_cost
                    ).reindex(evaluation.index)
                    test_sharpe = sharpe_ratio(evaluation.to_numpy())
                    alpha_pairs = final_holdout[["alpha_mean_12h", "target_12h"]].dropna()
                    alpha_ic = float(
                        alpha_pairs["alpha_mean_12h"].corr(
                            alpha_pairs["target_12h"], method="spearman"
                        )
                    )
                    artifact_path = str(Path(result.artifact_path).resolve())
                    final_candidates.append(
                        {
                            "seed": seed,
                            "validation_score": result.validation_score,
                            "test_sharpe": test_sharpe,
                            "alpha_ic_12h": alpha_ic,
                            "stress_return": float(np.prod(1 + stress.to_numpy()) - 1),
                            "artifact_path": artifact_path,
                            "training": training,
                            "evaluation": evaluation,
                            "stress": stress,
                            "training_benchmark": training_benchmark,
                            "evaluation_benchmark": evaluation_benchmark,
                            "training_deterministic": training_deterministic,
                            "evaluation_deterministic": evaluation_deterministic,
                            "evaluation_volatility_benchmark": evaluation_volatility_benchmark,
                            "feature_manifest": final_feature_manifest,
                            "feature_artifact_path": str(feature_path.resolve()),
                            "alpha_artifact_path": str(alpha_path.resolve()),
                        }
                    )
                    all_artifacts.append(artifact_path)
                    checkpoint_trials = max(1, len(result.validation_trials))
                    finite_trials = [
                        float(value) for value in result.validation_trials if np.isfinite(value)
                    ]
                    if finite_trials:
                        trial_sharpes.extend(finite_trials)
                    elif np.isfinite(result.validation_score):
                        trial_sharpes.append(float(result.validation_score))
                    trial_ledger.append(
                        {
                            "algorithm": champion_algorithm,
                            "fold": "sealed_holdout",
                            "seed": seed,
                            "checkpoint_trials": checkpoint_trials,
                            "validation_scores": list(result.validation_trials) or [result.validation_score],
                            "evaluation_sharpe": test_sharpe,
                            "stress_return": float(np.prod(1 + stress.to_numpy()) - 1),
                            "artifact_path": artifact_path,
                        }
                    )
                report_candidate = max(
                    final_candidates,
                    key=lambda item: float(item["validation_score"]),
                )
                report_candidate = {
                    **report_candidate,
                    "algorithm": champion_algorithm,
                    "fold": "sealed_holdout",
                }
                holdout_returns = report_candidate["evaluation"]
                holdout_seed_evaluation = SeedHarness(seeds).run(
                    lambda seed: {
                        "sharpe": sharpe_ratio(
                            next(
                                item["evaluation"]
                                for item in final_candidates
                                if item["seed"] == seed
                            ).to_numpy()
                        ),
                        "return": float(
                            np.prod(
                                1
                                + next(
                                    item["evaluation"]
                                    for item in final_candidates
                                    if item["seed"] == seed
                                ).to_numpy()
                            )
                            - 1
                        ),
                    }
                )
                final_metrics = institutional_metrics(
                    holdout_returns.to_numpy(),
                    trial_sharpes=trial_sharpes,
                    bootstrap_repetitions=2_000,
                )
                final_monte_carlo = moving_block_monte_carlo(
                    holdout_returns.to_numpy(),
                    paths=self.config.validation.monte_carlo_paths,
                    block_size=min(24, len(holdout_returns)),
                )
                final_metrics.update(
                    {
                        "ruin_probability_20": final_monte_carlo.ruin_probability_20,
                        "ruin_probability_30": final_monte_carlo.ruin_probability_30,
                        "buy_and_hold_return": float(
                            np.prod(1 + report_candidate["evaluation_benchmark"].to_numpy()) - 1
                        ),
                        "stress_return": float(
                            np.prod(1 + report_candidate["stress"].to_numpy()) - 1
                        ),
                        "deterministic_alpha_return": float(
                            np.prod(1 + report_candidate["evaluation_deterministic"].to_numpy()) - 1
                        ),
                        **pbo,
                    }
                )
                final_metrics["excess_return_vs_buy_and_hold"] = float(
                    np.prod(1 + holdout_returns.to_numpy()) - 1
                ) - final_metrics["buy_and_hold_return"]
                final_metrics["excess_return_vs_deterministic_alpha"] = float(
                    np.prod(1 + holdout_returns.to_numpy()) - 1
                ) - final_metrics["deterministic_alpha_return"]
                final_alpha_cash_test = paired_block_bootstrap_test(
                    report_candidate["evaluation_deterministic"].to_numpy(),
                    np.zeros(len(report_candidate["evaluation_deterministic"])),
                    seed=4,
                )
                final_metrics["deterministic_alpha_ci95_low"] = final_alpha_cash_test[
                    "ci_low"
                ]
                final_metrics["alpha_diagnostic_pass"] = bool(
                    report_candidate["feature_manifest"]["alpha"]["diagnostic_pass"]
                )
                development_metrics = algorithms[champion_algorithm]["metrics"]
                final_metrics["alpha_ic_mean"] = float(report_candidate["alpha_ic_12h"])
                final_metrics["alpha_ic_positive_fraction"] = development_metrics[
                    "alpha_ic_positive_fraction"
                ]
                final_alpha_test = paired_block_bootstrap_test(
                    holdout_returns.to_numpy(),
                    report_candidate["evaluation_deterministic"].reindex(
                        holdout_returns.index
                    ).to_numpy(),
                )
                final_volatility_test = paired_block_bootstrap_test(
                    holdout_returns.to_numpy(),
                    report_candidate["evaluation_volatility_benchmark"].reindex(
                        holdout_returns.index
                    ).to_numpy(),
                    seed=1,
                )
                final_confidence_set = model_confidence_set(
                    {
                        PUFFER_AGENT_NAME: holdout_returns.to_numpy(),
                        "Alpha": report_candidate["evaluation_deterministic"].reindex(
                            holdout_returns.index
                        ).to_numpy(),
                        "Vol B&H": report_candidate[
                            "evaluation_volatility_benchmark"
                        ].reindex(holdout_returns.index).to_numpy(),
                    },
                    seed=2,
                )
                final_metrics.update(
                    {
                        "paired_alpha_ci95_low": final_alpha_test["ci_low"],
                        "paired_volatility_bh_ci95_low": final_volatility_test["ci_low"],
                        "mcs_90_pass": bool(
                            final_confidence_set["retained"] == [PUFFER_AGENT_NAME]
                            and set(final_confidence_set["eliminated"])
                            == {"Alpha", "Vol B&H"}
                        ),
                        "model_confidence_set": final_confidence_set,
                        "seed_iqm_return_ci95_low": holdout_seed_evaluation.aggregate.get(
                            "return", {}
                        ).get("ci95_low", float("-inf")),
                    }
                )
                for metric in (
                    "profitable_fold_fraction",
                    "max_fold_profit_share",
                    "seed_stability_pass",
                ):
                    final_metrics[metric] = development_metrics[metric]
                final_eligible, final_reasons = promotion_gate(
                    final_metrics,
                    profile=self.config.profile,
                )
                report_candidate["eligible"] = final_eligible
                if final_eligible:
                    eligible_artifacts.append(str(report_candidate["artifact_path"]))
                final_evaluation = {
                    "eligible": final_eligible,
                    "gate_reasons": final_reasons,
                    "algorithm": champion_algorithm,
                    "selected_seed": report_candidate["seed"],
                    "selected_artifact": report_candidate["artifact_path"],
                    "feature_artifact_path": str(feature_path.resolve()),
                    "alpha_artifact_path": str(alpha_path.resolve()),
                    "training_range": [
                        str(report_candidate["training"].index.min()),
                        str(report_candidate["training"].index.max()),
                    ],
                    "holdout_range": [str(holdout_returns.index.min()), str(holdout_returns.index.max())],
                    "metrics": final_metrics,
                    "monte_carlo": final_monte_carlo.manifest(),
                    "seed_results": [
                        {
                            key: value
                            for key, value in candidate.items()
                            if key
                            not in {
                                "training",
                                "evaluation",
                                "stress",
                                "training_benchmark",
                                "evaluation_benchmark",
                                "training_deterministic",
                                "evaluation_deterministic",
                                "evaluation_volatility_benchmark",
                                "feature_manifest",
                                "feature_artifact_path",
                                "alpha_artifact_path",
                            }
                        }
                        for candidate in final_candidates
                    ],
                }
                algorithms[champion_algorithm]["eligible"] = final_eligible
                algorithms[champion_algorithm]["final_evaluation"] = final_evaluation

        eligible = bool(eligible_artifacts)
        reasons = (
            []
            if eligible
            else list(final_evaluation["gate_reasons"])
            if final_evaluation is not None
            else ["no algorithm passed every development promotion gate; sealed holdout was not opened"]
            if self.config.profile == "full"
            else ["no algorithm passed every promotion gate"]
        )
        artifact_bundle: dict[str, object] | None = None
        if final_evaluation is not None:
            model_path = Path(str(report_candidate["artifact_path"])).resolve()
            feature_artifact = Path(str(report_candidate["feature_artifact_path"])).resolve()
            alpha_artifact = Path(str(report_candidate["alpha_artifact_path"])).resolve()
            artifact_bundle = {
                "agent_name": PUFFER_AGENT_NAME,
                "model_path": str(model_path),
                "feature_pipeline_path": str(feature_artifact),
                "alpha_pipeline_path": str(alpha_artifact),
                "algorithm": report_candidate["algorithm"],
                "symbol": symbol,
                "book": symbol.lower().replace("/", "_"),
                "feature_order": list(report_candidate["feature_manifest"]["feature_order"]),
                "action_contract": "target_exposure_long_cash_v1",
                "observation_schema": "alpha_risk_state_v1",
                "market_context": (
                    "binance_public_v1"
                    if any(str(column).startswith("ctx_") for column in report_candidate["feature_manifest"]["feature_order"])
                    else "none"
                ),
                "decision_frequency": "1h",
                "execution_delay": "next_m1_tick",
                "feature_z_limit": 10.0,
                "minimum_shadow_days": 90,
                "commission_bps": self.config.validation.commission_bps,
                "base_spread_bps": self.config.validation.base_spread_bps,
                "max_risk_fraction": self.config.rl.risk_fractions[0],
                "sha256": {
                    "model": file_sha256(model_path),
                    "feature_pipeline": file_sha256(feature_artifact),
                    "alpha_pipeline": file_sha256(alpha_artifact),
                },
            }
        manifest: dict[str, object] = {
            "schema_version": 4,
            "agent_name": PUFFER_AGENT_NAME,
            "model_id": f"{symbol.replace('/', '_')}-{self._git_sha()[:12]}",
            "symbol": symbol,
            "profile": self.config.profile,
            "eligible": eligible,
            "gate_reasons": reasons,
            "selected_artifact": None,
            "artifact_paths": all_artifacts,
            "eligible_artifacts": eligible_artifacts,
            "config": self.config.public_dict(),
            "config_hash": self.config.config_hash,
            "data_hash": dataframe_hash(decision_data),
            "dependency_versions": dependency_versions(),
            "git_sha": self._git_sha(),
            "seeds": list(seeds),
            "algorithms": algorithms,
            "trial_count": len(trial_sharpes) + len(alpha_trial_ledger),
            "strategy_trial_count": len(trial_sharpes),
            "alpha_trial_count": len(alpha_trial_ledger),
            "trial_ledger": trial_ledger,
            "pbo": pbo,
            "validation_protocol": {
                "outer": "rolling" if self.config.profile == "full" else "single-purged-cpcv-smoke",
                "train_months": effective_train_months,
                "requested_train_months": (
                    self.config.validation.train_months if self.config.profile == "full" else None
                ),
                "validation_months": self.config.validation.validation_months if self.config.profile == "full" else None,
                "evaluation_months": self.config.validation.evaluation_months if self.config.profile == "full" else None,
                "step_months": self.config.validation.step_months if self.config.profile == "full" else None,
                "sealed_holdout": (
                    [str(holdout_start), str(holdout_end)]
                    if holdout_start is not None and holdout_end is not None
                    else None
                ),
            },
            "final_evaluation": final_evaluation,
            "artifact_bundle": artifact_bundle,
            "report_candidate": {
                "agent_name": PUFFER_AGENT_NAME,
                "algorithm": report_candidate["algorithm"],
                "fold": report_candidate["fold"],
                "seed": report_candidate["seed"],
                "artifact_path": report_candidate["artifact_path"],
                "validation_score": report_candidate["validation_score"],
            },
        }
        write_manifest(manifest, self.config.outputs_dir / f"{symbol.replace('/', '_')}_manifest.json")
        manifest["_reporting"] = {
            "agent_name": PUFFER_AGENT_NAME,
            "training": report_candidate["training"],
            "training_benchmark": report_candidate["training_benchmark"],
            "evaluation": report_candidate["evaluation"],
            "evaluation_benchmark": report_candidate["evaluation_benchmark"],
            "training_deterministic": report_candidate["training_deterministic"],
            "evaluation_deterministic": report_candidate["evaluation_deterministic"],
            "evaluation_volatility_benchmark": report_candidate["evaluation_volatility_benchmark"],
            "algorithm": report_candidate["algorithm"],
            "fold": report_candidate["fold"],
            "seed": report_candidate["seed"],
        }
        return manifest
