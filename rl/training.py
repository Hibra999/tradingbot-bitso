from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import AppConfig
from data import make_sliding_folds
from quant import CausalFeaturePipeline, atr
from validation import (
    CPCVSplitter,
    PerturbationConfig,
    SeedHarness,
    institutional_metrics,
    moving_block_monte_carlo,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

from .candidates import RecurrentPolicyRunner, build_cvar_qrdqn, build_recurrent_ppo, build_sac, build_tqc
from .environment import BracketTradingEnvV2
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


@dataclass(frozen=True)
class _ResearchFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    episode_segments: tuple[np.ndarray, ...]
    test_groups: tuple[object, ...]


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


class SB3CandidateRunner:
    """External-machine runner; validation selects a checkpoint, tests only score it."""

    def __init__(
        self,
        timesteps: int | dict[str, int] = 100_000,
        evaluations: int = 5,
        notifier: object | None = None,
        parallel_envs: int = 16,
    ):
        if evaluations < 1 or parallel_envs < 1:
            raise ValueError("evaluations and parallel_envs must be positive")
        self.timesteps = timesteps
        self.evaluations = evaluations
        self.notifier = notifier
        self.parallel_envs = parallel_envs

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
            action_mode={
                "recurrent_ppo": "ppo",
                "sac": "sac",
                "tqc": "sac",
                "cvar_qrdqn": "qrdqn",
            }[dataset.algorithm],
            randomize=randomize,
            random_seed=random_seed,
            allow_short=False,
            commission_rate=dataset.commission_rate,
            base_spread_bps=dataset.base_spread_bps * (
                dataset.stress_spread_multiplier if stress else 1.0
            ),
            perturbation_config=PerturbationConfig(
                slippage_atr_fraction=dataset.stress_slippage_atr_fraction
            ),
        )

    @staticmethod
    def _buy_and_hold(segments: tuple[pd.DataFrame, ...]) -> pd.Series:
        values = [segment["Close"].pct_change().dropna().astype(float) for segment in segments]
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
            recurrent = RecurrentPolicyRunner(model) if dataset.algorithm == "recurrent_ppo" else None
            values: list[float] = []
            timestamps: list[pd.Timestamp] = []
            done, previous_equity = False, environment.core.equity
            started, last = time.monotonic(), 0.0
            status = (
                f"{dataset.symbol} | {dataset.algorithm} | fold {dataset.fold + 1} | "
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
                    action = recurrent.predict(observation) if recurrent else model.predict(observation, deterministic=True)[0]
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

    def __call__(self, dataset: CandidateDataset) -> CandidateRun:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        notifier = self.notifier

        class ProgressCallback(BaseCallback):
            def __init__(self, bar, offset: int, steps: int, label: str, started: float):
                super().__init__(verbose=0)
                self.bar, self.offset, self.steps = bar, offset, steps
                self.label, self.started, self.base, self.last = label, started, 0, 0.0

            def _on_training_start(self) -> None:
                self.base = self.model.num_timesteps

            def _on_step(self) -> bool:
                current = self.offset + min(self.steps, max(self.model.num_timesteps - self.base, 0))
                if self.n_calls % 128 and current < self.offset + self.steps:
                    return True
                if current > self.bar.n:
                    self.bar.update(current - self.bar.n)
                now = time.monotonic()
                if now - self.last >= 1.0:
                    publish = getattr(notifier, "progress", None)
                    update = getattr(notifier, "update", None)
                    if publish:
                        publish(self.label, int(self.bar.n), int(self.bar.total), self.started)
                    elif update:
                        update(f"{self.label} | {int(self.bar.n):,}/{int(self.bar.total):,}")
                    self.last = now
                return True

        segments = dataset.training_segments
        if not segments:
            raise ValueError("training segments must not be empty")
        environment_count = (
            max(len(segments), self.parallel_envs) if dataset.algorithm == "recurrent_ppo" else len(segments)
        )
        factories = [
            lambda frame=segments[index % len(segments)], seed=dataset.seed + index: self._environment(
                dataset, frame, seed
            )
            for index in range(environment_count)
        ]
        training_env = DummyVecEnv(factories)
        builder = {
            "recurrent_ppo": build_recurrent_ppo,
            "sac": build_sac,
            "tqc": build_tqc,
            "cvar_qrdqn": build_cvar_qrdqn,
        }[dataset.algorithm]
        model = builder(training_env, seed=dataset.seed)
        total = self.timesteps.get(dataset.algorithm, 100_000) if isinstance(self.timesteps, dict) else self.timesteps
        chunk, remainder = divmod(total, self.evaluations)
        steps_by_evaluation = tuple(max(1, chunk + (evaluation < remainder)) for evaluation in range(self.evaluations))
        best_score = float("-inf")
        validation_scores: list[float] = []
        artifact = dataset.output_dir / "best_model"
        label = (
            f"{dataset.symbol} | {dataset.algorithm} | fold {dataset.fold + 1} | "
            f"seed {dataset.seed} | {environment_count} envs"
        )
        started = time.monotonic()
        with tqdm(
            total=sum(steps_by_evaluation),
            desc=f"{dataset.symbol} {dataset.algorithm} f{dataset.fold + 1} s{dataset.seed}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            for evaluation, steps in enumerate(steps_by_evaluation, 1):
                status = f"{label} | evaluation {evaluation}/{self.evaluations}"
                offset = int(progress.n)
                model.learn(
                    steps,
                    reset_num_timesteps=False,
                    callback=ProgressCallback(progress, offset, steps, status, started),
                )
                target = offset + steps
                if target > progress.n:
                    progress.update(target - progress.n)
                publish = getattr(notifier, "progress", None)
                if publish:
                    publish(status, int(progress.n), int(progress.total), started)
                update = getattr(notifier, "update", None)
                if update:
                    update(f"{status} | validation")
                validation_returns = self._evaluate(model, dataset, (dataset.validation_segment,))
                score = float(np.prod(1 + validation_returns.to_numpy()) - 1)
                validation_scores.append(score)
                if evaluation == 1 or np.isfinite(score) and (not np.isfinite(best_score) or score > best_score):
                    best_score = score
                    model.save(artifact)
                progress.set_postfix(
                    envs=environment_count,
                    evaluation=f"{evaluation}/{self.evaluations}",
                    best=f"{best_score:.4f}",
                    **_gpu_postfix(),
                )
        model = model.__class__.load(artifact, env=training_env)
        update = getattr(notifier, "update", None)
        if update:
            update(f"{label} | test evaluation")
        training_returns = self._evaluate(model, dataset, dataset.training_segments)
        test_returns = self._evaluate(model, dataset, dataset.test_segments)
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
            str(artifact.with_suffix(".zip")),
            best_score,
            tuple(str(value) for value in test_returns.index),
            tuple(training_returns.to_numpy()),
            tuple(str(value) for value in training_returns.index),
            tuple(self._buy_and_hold(dataset.test_segments).reindex(test_returns.index).to_numpy()),
            tuple(self._buy_and_hold(dataset.training_segments).reindex(training_returns.index).to_numpy()),
            tuple(validation_scores),
            tuple(stress_returns.to_numpy()),
            tuple(str(value) for value in stress_returns.index),
        )


class TrainingEngine:
    def __init__(
        self,
        config: AppConfig,
        runner: Callable[[CandidateDataset], CandidateRun] | None = None,
        notifier: object | None = None,
    ):
        self.config = config
        self.runner = runner or SB3CandidateRunner(
            config.rl.timesteps, config.rl.evaluations, notifier, config.rl.recurrent_ppo_envs
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
        holdout_start: pd.Timestamp | None = None
        holdout_end: pd.Timestamp | None = None
        folds: list[_ResearchFold] = []
        if self.config.profile == "full":
            latest = decision_data.index.max()
            holdout_end = pd.Timestamp(latest.year, latest.month, 1, tz=latest.tz)
            holdout_start = holdout_end - pd.DateOffset(months=self.config.validation.holdout_months)
            development = decision_data.loc[decision_data.index < holdout_start]
            windows = make_sliding_folds(
                development,
                train_years=self.config.validation.train_months / 12,
                val_months=self.config.validation.validation_months,
                test_months=self.config.validation.evaluation_months,
                step_months=self.config.validation.step_months,
                embargo_bars=self.config.validation.embargo_bars,
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
            if len(folds) < 2:
                raise ValueError("full validation requires at least two complete walk-forward folds")
        else:
            splitter = CPCVSplitter(
                self.config.validation.temporal_groups,
                self.config.validation.test_groups,
                self.config.validation.embargo_bars,
                holding,
            )
            for fold in splitter.split(decision_data.index, interval_end):
                train_indices, validation_indices = internal_purged_validation_tail(
                    fold.train_indices,
                    decision_data.index,
                    interval_end,
                    embargo_bars=self.config.validation.embargo_bars,
                )
                folds.append(
                    _ResearchFold(
                        train_indices,
                        validation_indices,
                        fold.episode_segments,
                        tuple(fold.test_groups),
                    )
                )
        seeds = self.config.validation.full_seeds if self.config.profile == "full" else self.config.validation.smoke_seeds
        algorithms: dict[str, object] = {}
        all_artifacts: list[str] = []
        eligible_artifacts: list[str] = []
        all_prior = np.arange(len(decision_data))
        prepared = []
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
            pipeline = CausalFeaturePipeline(config_hash=self.config.config_hash, random_state=fold_number)
            pipeline.fit(decision_data.iloc[train_indices])

            def transformed(indices: np.ndarray, history_indices: np.ndarray) -> pd.DataFrame:
                raw = decision_data.iloc[indices]
                prior = history_indices[history_indices < indices.min()]
                features = pipeline.transform(raw, history_context=decision_data.iloc[prior])
                return raw.loc[features.index].join(features)

            prepared.append(
                (
                    fold_number,
                    fold,
                    pipeline.feature_order,
                    tuple(transformed(segment, train_indices) for segment in _contiguous(train_indices)),
                    transformed(validation_indices, train_indices),
                    tuple(transformed(segment, all_prior) for segment in fold.episode_segments),
                    pipeline.manifest(),
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
        trial_ledger: list[dict[str, object]] = []
        for algorithm in tqdm(
            self.config.rl.algorithms,
            desc=f"{symbol} algorithms",
            leave=False,
            dynamic_ncols=True,
        ):
            candidates: list[dict[str, object]] = []
            jobs = tqdm(
                total=len(prepared) * len(seeds),
                desc=f"{algorithm} fold/seed jobs",
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
                    test_sharpe = sharpe_ratio(evaluation.to_numpy()) if len(evaluation) > 1 else float("nan")
                    artifact_path = str(Path(result.artifact_path).resolve())
                    candidate = {
                        "fold": fold_number,
                        "seed": seed,
                        "test_groups": list(fold.test_groups),
                        "validation_score": result.validation_score,
                        "test_sharpe": test_sharpe,
                        "stress_return": float(np.prod(1 + stress.to_numpy()) - 1),
                        "artifact_path": artifact_path,
                        "feature_manifest": feature_manifest,
                        "evaluation": evaluation,
                        "stress": stress,
                        "training": training,
                        "evaluation_benchmark": evaluation_benchmark,
                        "training_benchmark": training_benchmark,
                    }
                    candidates.append(candidate)
                    all_artifacts.append(artifact_path)
                    trial_count = max(1, len(result.validation_trials))
                    trial_sharpes.extend([test_sharpe] * trial_count)
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
                            f"{symbol} | {algorithm} | fold {fold_number + 1}/{len(folds)} | "
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
            seed_validation_scores: dict[int, float] = {}
            for seed, items in seed_candidates.items():
                if self.config.profile == "full":
                    path = pd.concat([item["evaluation"] for item in items]).sort_index()
                    stress_path = pd.concat([item["stress"] for item in items]).sort_index()
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
                    validation_score = float(representative["validation_score"])
                seed_paths[seed] = path
                seed_stress_paths[seed] = stress_path
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
                paths=self.config.validation.monte_carlo_paths,
                block_size=min(24, len(evaluation)),
            )
            metrics["ruin_probability_20"] = monte_carlo.ruin_probability_20
            metrics["ruin_probability_30"] = monte_carlo.ruin_probability_30
            strategy_return = float(np.prod(1 + evaluation.to_numpy()) - 1)
            buy_and_hold_return = float(np.prod(1 + evaluation_benchmark.to_numpy()) - 1)
            metrics["buy_and_hold_return"] = buy_and_hold_return
            metrics["excess_return_vs_buy_and_hold"] = strategy_return - buy_and_hold_return
            metrics["stress_return"] = float(
                np.prod(1 + seed_stress_paths[selected_seed].to_numpy()) - 1
            )
            metrics.update(pbo)
            algorithm_eligible, algorithm_reasons = promotion_gate(metrics, profile=self.config.profile)
            if algorithm_eligible and self.config.profile != "full":
                eligible_artifacts.append(str(selected["artifact_path"]))
            notify = getattr(self.notifier, "notify", None)
            if notify:
                notify(
                    f"{symbol} | {algorithm} | {'PASS' if algorithm_eligible else 'FAIL'} | "
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
                    "selected_seed": selected_seed,
                }
            )
            algorithms[algorithm] = {
                "eligible": algorithm_eligible if self.config.profile != "full" else False,
                "development_eligible": algorithm_eligible,
                "gate_reasons": algorithm_reasons,
                "selected_artifact": selected["artifact_path"] if self.config.profile != "full" else None,
                "selection_rule": "highest mean validation return across chronological folds",
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
                            "feature_manifest",
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
                    return raw.loc[features.index].join(features)

                final_training = final_transformed(final_train_indices, final_train_indices)
                final_validation = final_transformed(final_validation_indices, final_train_indices)
                final_holdout = final_transformed(final_holdout_indices, all_prior)
                feature_path = self.config.models_dir / symbol.replace("/", "_") / "final" / "features.pkl"
                final_pipeline.save(feature_path)
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
                        final_pipeline.feature_order,
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
                    test_sharpe = sharpe_ratio(evaluation.to_numpy())
                    artifact_path = str(Path(result.artifact_path).resolve())
                    final_candidates.append(
                        {
                            "seed": seed,
                            "validation_score": result.validation_score,
                            "test_sharpe": test_sharpe,
                            "stress_return": float(np.prod(1 + stress.to_numpy()) - 1),
                            "artifact_path": artifact_path,
                            "training": training,
                            "evaluation": evaluation,
                            "stress": stress,
                            "training_benchmark": training_benchmark,
                            "evaluation_benchmark": evaluation_benchmark,
                            "feature_manifest": final_pipeline.manifest(),
                            "feature_artifact_path": str(feature_path.resolve()),
                        }
                    )
                    all_artifacts.append(artifact_path)
                    checkpoint_trials = max(1, len(result.validation_trials))
                    trial_sharpes.extend([test_sharpe] * checkpoint_trials)
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
                        **pbo,
                    }
                )
                final_metrics["excess_return_vs_buy_and_hold"] = float(
                    np.prod(1 + holdout_returns.to_numpy()) - 1
                ) - final_metrics["buy_and_hold_return"]
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
                                "feature_manifest",
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
            artifact_bundle = {
                "model_path": str(model_path),
                "feature_pipeline_path": str(feature_artifact),
                "algorithm": report_candidate["algorithm"],
                "symbol": symbol,
                "book": symbol.lower().replace("/", "_"),
                "feature_order": list(report_candidate["feature_manifest"]["feature_order"]),
                "action_contract": "long_flat_spot",
                "decision_frequency": "1h",
                "execution_delay": "next_m1_tick",
                "feature_z_limit": 10.0,
                "commission_bps": self.config.validation.commission_bps,
                "base_spread_bps": self.config.validation.base_spread_bps,
                "sha256": {
                    "model": file_sha256(model_path),
                    "feature_pipeline": file_sha256(feature_artifact),
                },
            }
        manifest: dict[str, object] = {
            "schema_version": 2,
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
            "trial_count": len(trial_sharpes),
            "trial_ledger": trial_ledger,
            "pbo": pbo,
            "validation_protocol": {
                "outer": "rolling" if self.config.profile == "full" else "cpcv-smoke",
                "train_months": self.config.validation.train_months if self.config.profile == "full" else None,
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
                "algorithm": report_candidate["algorithm"],
                "fold": report_candidate["fold"],
                "seed": report_candidate["seed"],
                "artifact_path": report_candidate["artifact_path"],
                "validation_score": report_candidate["validation_score"],
            },
        }
        write_manifest(manifest, self.config.outputs_dir / f"{symbol.replace('/', '_')}_manifest.json")
        manifest["_reporting"] = {
            "training": report_candidate["training"],
            "training_benchmark": report_candidate["training_benchmark"],
            "evaluation": report_candidate["evaluation"],
            "evaluation_benchmark": report_candidate["evaluation_benchmark"],
            "algorithm": report_candidate["algorithm"],
            "fold": report_candidate["fold"],
            "seed": report_candidate["seed"],
        }
        return manifest
