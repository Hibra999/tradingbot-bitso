from __future__ import annotations

import subprocess
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
    """Return GPU memory info dict for tqdm postfix, or empty dict if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {"GPU_alloc": f"{allocated:.1f}GB", "GPU_rsv": f"{reserved:.1f}GB"}
    except Exception:
        pass
    return {}


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


class SB3CandidateRunner:
    """External-machine runner; validation selects a checkpoint, tests only score it."""

    def __init__(self, timesteps: int | dict[str, int] = 100_000, evaluations: int = 10):
        self.timesteps = timesteps
        self.evaluations = evaluations

    @staticmethod
    def _environment(dataset: CandidateDataset, frame: pd.DataFrame) -> BracketTradingEnvV2:
        return BracketTradingEnvV2(
            frame,
            dataset.m1_bars,
            list(dataset.feature_columns),
            action_mode={"recurrent_ppo": "ppo", "sac": "sac", "cvar_qrdqn": "qrdqn"}[dataset.algorithm],
            randomize=True,
        )

    @staticmethod
    def _evaluate(model, dataset: CandidateDataset, segments: tuple[pd.DataFrame, ...]) -> tuple[float, ...]:
        returns: list[float] = []
        for seg_idx, segment in enumerate(segments):
            environment = SB3CandidateRunner._environment(dataset, segment)
            observation, _ = environment.reset(seed=dataset.seed)
            recurrent = RecurrentPolicyRunner(model) if dataset.algorithm == "recurrent_ppo" else None
            done, previous_equity = False, environment.core.equity
            step_count = 0
            pbar = tqdm(desc=f"Eval seg {seg_idx + 1}/{len(segments)}", unit="step", leave=False, mininterval=0.5)
            while not done:
                action = recurrent.predict(observation) if recurrent else model.predict(observation, deterministic=True)[0]
                observation, _, done, _, info = environment.step(action)
                equity = max(float(info["equity"]), 0.0)
                ret = (equity / previous_equity - 1.0) if previous_equity > 0 else 0.0
                returns.append(max(ret, -1.0))
                previous_equity = equity
                step_count += 1
                pbar.update(1)
            pbar.close()
        return tuple(returns)

    def __call__(self, dataset: CandidateDataset) -> CandidateRun:
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Log GPU device once
        try:
            import torch
            if torch.cuda.is_available():
                dev_name = torch.cuda.get_device_name(0)
                tqdm.write(f"  GPU: {dev_name}")
        except Exception:
            pass

        factories = [lambda frame=frame: self._environment(dataset, frame) for frame in dataset.training_segments]
        training_env = DummyVecEnv(factories)
        builder = {
            "recurrent_ppo": build_recurrent_ppo,
            "sac": build_sac,
            "cvar_qrdqn": build_cvar_qrdqn,
        }[dataset.algorithm]
        model = builder(training_env, seed=dataset.seed)
        total_ts = self.timesteps.get(dataset.algorithm, 100_000) if isinstance(self.timesteps, dict) else self.timesteps
        chunk = max(1, total_ts // self.evaluations)
        best_score = float("-inf")
        artifact = dataset.output_dir / "best_model"
        training_bar = tqdm(
            range(self.evaluations),
            desc=f"    Training {dataset.algorithm}",
            unit="chunk",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        for eval_idx in training_bar:
            model.learn(chunk, reset_num_timesteps=False)
            gpu_info = _gpu_postfix()
            postfix = {"chunk_ts": f"{chunk:,}", "best": f"{best_score:.4f}"}
            postfix.update(gpu_info)
            training_bar.set_postfix(postfix)
            validation_returns = self._evaluate(model, dataset, (dataset.validation_segment,))
            score = float(np.prod(1 + np.asarray(validation_returns)) - 1)
            if score > best_score:
                best_score = score
                model.save(artifact)
        training_bar.close()
        model = model.__class__.load(artifact, env=training_env)
        test_returns = self._evaluate(model, dataset, dataset.test_segments)
        return CandidateRun(test_returns, str(artifact.with_suffix(".zip")), best_score)


class TrainingEngine:
    def __init__(
        self,
        config: AppConfig,
        runner: Callable[[CandidateDataset], CandidateRun] | None = None,
        notifier: object | None = None,
    ):
        self.config = config
        self.runner = runner or SB3CandidateRunner()
        self._notifier = notifier

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

        for algorithm in tqdm(self.config.rl.algorithms, desc="Algorithms", unit="algo", leave=False):
            runs: list[dict[str, object]] = []
            algorithm_artifacts: list[str] = []
            returns_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
            feature_manifests: list[dict[str, object]] = []
            total_jobs = len(seeds) * len(folds)
            job_idx = 0
            fold_seed_bar = tqdm(
                total=total_jobs,
                desc=f"  {algorithm}",
                unit="job",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            for fold_number, fold in enumerate(folds):
                train_indices, validation_indices = internal_purged_validation_tail(
                    fold.train_indices,
                    decision_data.index,
                    interval_end,
                    embargo_bars=self.config.validation.embargo_bars,
                )
                pipeline = CausalFeaturePipeline(config_hash=self.config.config_hash, random_state=fold_number)
                training_segments_raw = tuple(decision_data.iloc[segment] for segment in _contiguous(train_indices))
                pipeline.fit(training_segments_raw)
                feature_manifests.append(pipeline.manifest())

                all_prior = np.arange(len(decision_data))

                def transformed(indices: np.ndarray, history_indices: np.ndarray) -> pd.DataFrame:
                    raw = decision_data.iloc[indices]
                    prior = history_indices[history_indices < indices.min()]
                    history = decision_data.iloc[prior]
                    features = pipeline.transform(raw, history_context=history)
                    return raw.loc[features.index].join(features)

                training_segments = tuple(transformed(segment, all_prior) for segment in _contiguous(train_indices))
                validation_segment = transformed(validation_indices, all_prior)
                test_segments = tuple(transformed(segment, all_prior) for segment in fold.episode_segments)
                for seed in seeds:
                    job_idx += 1
                    fold_seed_bar.set_postfix({"fold": f"{fold_number + 1}/{len(folds)}", "seed": seed})
                    output_dir = self.config.models_dir / symbol.replace("/", "_") / algorithm / f"fold_{fold_number}" / f"seed_{seed}"
                    dataset = CandidateDataset(
                        symbol,
                        algorithm,
                        seed,
                        fold_number,
                        pipeline.feature_order,
                        training_segments,
                        validation_segment,
                        test_segments,
                        m1_data,
                        output_dir,
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result = self.runner(dataset)
                    returns_by_seed[seed].extend(result.returns)
                    all_artifacts.append(result.artifact_path)
                    algorithm_artifacts.append(result.artifact_path)
                    runs.append(
                        {
                            "fold": fold_number,
                            "seed": seed,
                            "test_groups": list(fold.test_groups),
                            "validation_score": result.validation_score,
                            "test_sharpe": sharpe_ratio(result.returns) if len(result.returns) > 1 else float("nan"),
                            "artifact_path": result.artifact_path,
                        }
                    )
                    fold_seed_bar.update(1)
            fold_seed_bar.close()

            seed_sharpes = [sharpe_ratio(values) for values in returns_by_seed.values()]
            combined = np.concatenate([np.asarray(values, dtype=float) for values in returns_by_seed.values() if len(values)])
            trial_sharpes = [s for s in seed_sharpes if np.isfinite(s)]
            if len(seeds) == 1 or len(trial_sharpes) < 2:
                trial_sharpes = [float(run["test_sharpe"]) for run in runs if np.isfinite(float(run["test_sharpe"]))]
            if len(trial_sharpes) < 2:
                trial_sharpes = [0.0, 0.0]
            metrics = institutional_metrics(
                combined,
                trial_sharpes=trial_sharpes,
                bootstrap_repetitions=2_000 if self.config.profile == "full" else 100,
            )
            seed_evaluation = SeedHarness(seeds, smoke=self.config.profile == "smoke").run(
                lambda seed: {
                    "sharpe": sharpe_ratio(returns_by_seed[seed]),
                    "dsr_z": metrics["dsr_z"],
                    "return": float(np.prod(1 + np.asarray(returns_by_seed[seed])) - 1),
                }
            )
            monte_carlo = moving_block_monte_carlo(
                combined,
                paths=self.config.validation.monte_carlo_paths if self.config.profile == "full" else 100,
                block_size=min(24, len(combined)),
            )
            metrics["ruin_probability_20"] = monte_carlo.ruin_probability_20
            metrics["ruin_probability_30"] = monte_carlo.ruin_probability_30
            algorithm_eligible, algorithm_reasons = promotion_gate(metrics, profile=self.config.profile)
            if algorithm_eligible:
                eligible_artifacts.extend(algorithm_artifacts)

            # Notify Telegram with algorithm results
            if self._notifier and hasattr(self._notifier, "notify"):
                sharpe_val = metrics.get("sharpe", float("nan"))
                status = "PASS" if algorithm_eligible else "FAIL"
                reasons_str = ", ".join(algorithm_reasons) if algorithm_reasons else "--"
                self._notifier.notify(
                    f"  {algorithm}: {status}\n"
                    f"  Sharpe: {sharpe_val:.3f}\n"
                    f"  Reasons: {reasons_str}"
                )

            algorithms[algorithm] = {
                "eligible": algorithm_eligible,
                "gate_reasons": algorithm_reasons,
                "fold_results": runs,
                "seed_evaluation": seed_evaluation.manifest(),
                "metrics": metrics,
                "monte_carlo": monte_carlo.manifest(),
                "feature_manifests": feature_manifests,
            }

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
        }
        write_manifest(manifest, self.config.outputs_dir / f"{symbol.replace('/', '_')}_manifest.json")
        return manifest
