from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from config import env_flag

_PACKAGES = ("numpy", "pandas", "pufferlib", "torch", "gymnasium")


def dataframe_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    result = {}
    for package in _PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def promotion_gate(
    metrics: dict[str, Any],
    *,
    profile: str,
    max_drawdown_limit: float = -0.20,
    ruin_probability_20_limit: float = 0.05,
    pbo_limit: float = 0.10,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if profile != "full":
        reasons.append("smoke profiles are non-promotable")
    if metrics.get("dsr_p_value", 1.0) >= 0.05:
        reasons.append("DSR p-value must be below 0.05")
    if metrics.get("max_drawdown", -1.0) < max_drawdown_limit:
        reasons.append("maximum drawdown gate failed")
    if metrics.get("ruin_probability_20", 1.0) > ruin_probability_20_limit:
        reasons.append("20% ruin-probability gate failed")
    if metrics.get("pbo_probability", 1.0) > pbo_limit:
        reasons.append("probability of backtest overfitting must not exceed 10%")
    if metrics.get("excess_return_vs_buy_and_hold", float("-inf")) <= 0:
        reasons.append("net return must exceed buy-and-hold on identical timestamps")
    if metrics.get("excess_return_vs_deterministic_alpha", float("-inf")) <= 0:
        reasons.append("PuffeRL-LSTM must exceed the deterministic alpha policy")
    if metrics.get("deterministic_alpha_return", float("-inf")) <= 0:
        reasons.append("deterministic alpha must remain profitable after costs")
    if metrics.get("deterministic_alpha_ci95_low", float("-inf")) <= 0:
        reasons.append("deterministic alpha lower confidence bound must be positive")
    if not bool(metrics.get("alpha_diagnostic_pass", False)):
        reasons.append("alpha experts did not beat mean and permuted-target controls")
    if metrics.get("alpha_ic_mean", float("-inf")) <= 0:
        reasons.append("12h alpha information coefficient must be positive")
    if metrics.get("alpha_ic_positive_fraction", 0.0) < 0.70:
        reasons.append("12h alpha information coefficient must be positive in at least 70% of folds")
    if metrics.get("stress_return", float("-inf")) <= 0:
        reasons.append("stressed execution return must remain positive")
    if metrics.get("bootstrap_mean_ci95_low", float("-inf")) <= 0:
        reasons.append("95% block-bootstrap mean-return lower bound must be positive")
    if metrics.get("paired_alpha_ci95_low", float("-inf")) <= 0:
        reasons.append("paired 95% lower bound versus deterministic alpha must be positive")
    if metrics.get("paired_volatility_bh_ci95_low", float("-inf")) <= 0:
        reasons.append("paired 95% lower bound versus volatility-matched buy-and-hold must be positive")
    if not bool(metrics.get("mcs_90_pass", False)):
        reasons.append("90% model confidence set did not eliminate both production baselines")
    if metrics.get("seed_iqm_return_ci95_low", float("-inf")) <= 0:
        reasons.append("five-seed IQM return lower confidence bound must be positive")
    if not bool(metrics.get("seed_stability_pass", False)):
        reasons.append("seed stability gate failed")
    if metrics.get("profitable_fold_fraction", 0.0) < 0.70:
        reasons.append("at least 70% of evaluation folds must be profitable")
    if metrics.get("max_fold_profit_share", 1.0) > 0.50:
        reasons.append("one evaluation fold contributes more than half of positive profit")
    return not reasons, reasons


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return destination


def load_eligible_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4 or manifest.get("profile") != "full" or not manifest.get("eligible"):
        raise PermissionError("manifest schema/profile/gates do not permit live loading")
    selected = manifest.get("selected_artifact")
    declared = set(manifest.get("artifact_paths", []))
    eligible_artifacts = set(manifest.get("eligible_artifacts", []))
    if not selected or selected not in declared or selected not in eligible_artifacts:
        raise PermissionError("an operator must select one declared, gate-eligible artifact")
    bundle = manifest.get("artifact_bundle")
    if not isinstance(bundle, dict) or bundle.get("model_path") != selected:
        raise PermissionError("selected artifact must match the complete champion bundle")
    if bundle.get("algorithm") != "pufferl":
        raise PermissionError("champion bundle must use PuffeRL-LSTM")
    model_path = Path(selected).resolve()
    feature_path = Path(str(bundle.get("feature_pipeline_path", ""))).resolve()
    alpha_path = Path(str(bundle.get("alpha_pipeline_path", ""))).resolve()
    hashes = bundle.get("sha256", {})
    if not isinstance(hashes, dict):
        raise PermissionError("champion bundle hashes are invalid")
    if not model_path.is_file() or not feature_path.is_file() or not alpha_path.is_file():
        raise PermissionError("champion model, feature pipeline, and alpha pipeline must exist")
    if (
        hashes.get("model") != file_sha256(model_path)
        or hashes.get("feature_pipeline") != file_sha256(feature_path)
        or hashes.get("alpha_pipeline") != file_sha256(alpha_path)
    ):
        raise PermissionError("champion bundle hash verification failed")
    return manifest


def load_approved_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = load_eligible_manifest(manifest_path)
    if not env_flag("MODEL_APPROVED") or not env_flag("BITSO_LIVE_ENABLED"):
        raise PermissionError("MODEL_APPROVED=true and BITSO_LIVE_ENABLED=true are both required")
    approved_path = os.getenv("APPROVED_MODEL_MANIFEST", "").strip()
    if not approved_path or Path(approved_path).resolve() != manifest_path:
        raise PermissionError("APPROVED_MODEL_MANIFEST must name this exact manifest")
    return manifest
