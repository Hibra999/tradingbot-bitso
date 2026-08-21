from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from config import env_flag

_PACKAGES = ("numpy", "pandas", "stable-baselines3", "sb3-contrib", "torch", "gymnasium")


def dataframe_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def dependency_versions() -> dict[str, str]:
    result = {}
    for package in _PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def promotion_gate(
    metrics: dict[str, float],
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
    return not reasons, reasons


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return destination


def load_approved_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("profile") != "full" or not manifest.get("eligible"):
        raise PermissionError("manifest schema/profile/gates do not permit live loading")
    selected = manifest.get("selected_artifact")
    declared = set(manifest.get("artifact_paths", []))
    eligible_artifacts = set(manifest.get("eligible_artifacts", []))
    if not selected or selected not in declared or selected not in eligible_artifacts:
        raise PermissionError("an operator must select one declared, gate-eligible artifact")
    if not env_flag("MODEL_APPROVED") or not env_flag("BITSO_LIVE_ENABLED"):
        raise PermissionError("MODEL_APPROVED=true and BITSO_LIVE_ENABLED=true are both required")
    approved_path = os.getenv("APPROVED_MODEL_MANIFEST", "").strip()
    if not approved_path or Path(approved_path).resolve() != manifest_path:
        raise PermissionError("APPROVED_MODEL_MANIFEST must name this exact manifest")
    return manifest
