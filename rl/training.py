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
from quant import CausalFeaturePipeline, atr
from validation import (
    CPCVSplitter,
    SeedHarness,
    institutional_metrics,
    moving_block_monte_carlo,
    sharpe_ratio,
)

from .candidates import RecurrentPolicyRunner, build_cvar_qrdqn, build_recurrent_ppo, build_sac
from .environment import BracketTradingEnvV2
from .governance import dataframe_hash, dependency_versions, promotion_gate, write_manifest


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
    ) -> BracketTradingEnvV2:
        return BracketTradingEnvV2(
            frame,
            dataset.m1_bars,
            list(dataset.feature_columns),
            action_mode={"recurrent_ppo": "ppo", "sac": "sac", "cvar_qrdqn": "qrdqn"}[dataset.algorithm],
            randomize=randomize,
            random_seed=random_seed,
            allow_short=False,
        )

    @staticmethod
    def _buy_and_hold(segments: tuple[pd.DataFrame, ...]) -> pd.Series:
        values = [segment["Close"].pct_change().dropna().astype(float) for segment in segments]
        return pd.concat(values).sort_index() if values else pd.Series(dtype=float)

    def _evaluate(self, model, dataset: CandidateDataset, segments: tuple[pd.DataFrame, ...]) -> pd.Series:
        returns: list[pd.Series] = []
        for number, segment in enumerate(segments, 1):
            environment = self._environment(dataset, segment, randomize=False)
            observation, _ = environment.reset(seed=dataset.seed)
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
        decision_data = decision_data.copy()
        if "atr" not in decision_data:
            decision_data["atr"] = atr(decision_data)
        holding = self.config.validation.max_holding_bars
        end_positions = np.minimum(np.arange(len(decision_data)) + holding, len(decision_data) - 1)
        interval_end = pd.DatetimeIndex(decision_data.index[end_positions])
        splitter = CPCVSplitter(
            self.config.validation.temporal_groups,
            self.config.validation.test_groups,
            self.config.validation.embargo_bars,
            holding,
        )
        folds = splitter.split(decision_data.index, interval_end)
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
            train_indices, validation_indices = internal_purged_validation_tail(
                fold.train_indices,
                decision_data.index,
                interval_end,
                embargo_bars=self.config.validation.embargo_bars,
            )
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
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result = self.runner(dataset)
                    evaluation = _run_series(result, test_segments)
                    training = _run_series(result, training_segments, training=True)
                    evaluation_benchmark = _benchmark_series(result, evaluation, test_segments)
                    training_benchmark = _benchmark_series(result, training, training_segments, training=True)
                    test_sharpe = sharpe_ratio(evaluation.to_numpy()) if len(evaluation) > 1 else float("nan")
                    candidate = {
                        "fold": fold_number,
                        "seed": seed,
                        "test_groups": list(fold.test_groups),
                        "validation_score": result.validation_score,
                        "test_sharpe": test_sharpe,
                        "artifact_path": result.artifact_path,
                        "feature_manifest": feature_manifest,
                        "evaluation": evaluation,
                        "training": training,
                        "evaluation_benchmark": evaluation_benchmark,
                        "training_benchmark": training_benchmark,
                    }
                    candidates.append(candidate)
                    all_artifacts.append(result.artifact_path)
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
                            "artifact_path": result.artifact_path,
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

        report_candidates: list[dict[str, object]] = []
        for algorithm, candidates in candidates_by_algorithm.items():
            selected = max(candidates, key=lambda item: float(item["validation_score"]))
            evaluation = selected["evaluation"]
            if not isinstance(evaluation, pd.Series):
                raise TypeError("candidate evaluation must be a pandas Series")
            metrics = institutional_metrics(
                evaluation.to_numpy(),
                trial_sharpes=trial_sharpes,
                bootstrap_repetitions=2_000 if self.config.profile == "full" else 100,
            )
            seed_candidates = {
                seed: max(
                    (item for item in candidates if item["seed"] == seed),
                    key=lambda item: float(item["validation_score"]),
                )
                for seed in seeds
            }
            seed_evaluation = SeedHarness(seeds, smoke=self.config.profile == "smoke").run(
                lambda seed: {
                    "sharpe": float(seed_candidates[seed]["test_sharpe"]),
                    "dsr_z": metrics["dsr_z"],
                    "return": float(
                        np.prod(1 + seed_candidates[seed]["evaluation"].to_numpy()) - 1
                    ),
                }
            )
            monte_carlo = moving_block_monte_carlo(
                evaluation.to_numpy(),
                paths=self.config.validation.monte_carlo_paths,
                block_size=min(24, len(evaluation)),
            )
            metrics["ruin_probability_20"] = monte_carlo.ruin_probability_20
            metrics["ruin_probability_30"] = monte_carlo.ruin_probability_30
            algorithm_eligible, algorithm_reasons = promotion_gate(metrics, profile=self.config.profile)
            if algorithm_eligible:
                eligible_artifacts.append(str(selected["artifact_path"]))
            notify = getattr(self.notifier, "notify", None)
            if notify:
                notify(
                    f"{symbol} | {algorithm} | {'PASS' if algorithm_eligible else 'FAIL'} | "
                    f"Sharpe {metrics['sharpe']:.4f} | SQN {metrics['sqn']:.4f}"
                )
            report_candidates.append({**selected, "eligible": algorithm_eligible, "algorithm": algorithm})
            algorithms[algorithm] = {
                "eligible": algorithm_eligible,
                "gate_reasons": algorithm_reasons,
                "selected_artifact": selected["artifact_path"],
                "selection_rule": "highest validation return; evaluation never selects checkpoints",
                "fold_results": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in {
                            "evaluation",
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

        report_candidate = max(
            (item for item in report_candidates if item["eligible"]),
            key=lambda item: float(item["validation_score"]),
            default=max(report_candidates, key=lambda item: float(item["validation_score"])),
        )

        eligible = bool(eligible_artifacts)
        reasons = [] if eligible else ["no algorithm passed every promotion gate"]
        manifest: dict[str, object] = {
            "schema_version": 1,
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
