from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import AppConfig

from .features import add_range_volatility_features, atr
from .fracdiff import ADFSelection, fixed_width_fracdiff, select_adf_d
from .regimes import CausalRegimeModel
from .wavelets import rolling_wavelet_features

_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "statsmodels", "hmmlearn", "PyWavelets")


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for dependency in _DEPENDENCIES:
        try:
            result[dependency] = version(dependency)
        except PackageNotFoundError:
            result[dependency] = "unavailable"
    return result


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class CausalFeaturePipeline:
    def __init__(
        self,
        *,
        config_hash: str | None = None,
        fracdiff_threshold: float = 1e-5,
        wavelet_window: int = 256,
        random_state: int = 0,
    ):
        self.config_hash = config_hash or AppConfig().config_hash
        self.fracdiff_threshold = fracdiff_threshold
        self.wavelet_window = wavelet_window
        self.regime_model = CausalRegimeModel(random_state=random_state)
        self.scaler = StandardScaler()
        self.fracdiff_selection: ADFSelection | None = None
        self.feature_order: tuple[str, ...] = ()
        self.dataset_range: tuple[str, str] = ()

    def _deterministic(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fracdiff_selection is None:
            raise RuntimeError("FracDiff parameter is not fitted")
        out = add_range_volatility_features(frame, 20)
        out["log_return"] = np.log(out["Close"]).diff()
        out["trend_20"] = np.log(out["Close"] / out["Close"].ewm(span=20, adjust=False).mean())
        out["atr_close"] = atr(out, 14) / out["Close"]
        out["fracdiff_close"] = fixed_width_fracdiff(
            np.log(out["Close"]), self.fracdiff_selection.d, self.fracdiff_threshold
        )
        return out.join(
            rolling_wavelet_features(np.log(out["Close"]), window=self.wavelet_window, wavelet="db4", level=3)
        )

    @staticmethod
    def _candidate_columns(frame: pd.DataFrame) -> tuple[str, ...]:
        core = (
            "log_return",
            "trend_20",
            "atr_close",
            "parkinson_20",
            "garman_klass_20",
            "yang_zhang_20",
            "fracdiff_close",
            "wavelet_a3_endpoint",
            "wavelet_d3_energy",
            "wavelet_d2_energy",
            "wavelet_d1_energy",
            "regime_bear",
            "regime_neutral",
            "regime_bull",
        )
        live_excluded = {"obi", "micro_price"}
        realized = tuple(column for column in frame.columns if (column.startswith("rv_") or column.startswith("return_")) and column not in live_excluded)
        return core + realized

    def _unscaled(self, frame: pd.DataFrame) -> pd.DataFrame:
        deterministic = self._deterministic(frame)
        hmm_inputs = deterministic[["log_return", "trend_20", "garman_klass_20"]]
        return deterministic.join(self.regime_model.forward_probabilities(hmm_inputs))

    def fit(self, train_data: pd.DataFrame) -> "CausalFeaturePipeline":
        if train_data.empty or not train_data.index.is_monotonic_increasing:
            raise ValueError("training data must be non-empty and chronological")
        self.fracdiff_selection = select_adf_d(
            np.log(train_data["Close"]), threshold=self.fracdiff_threshold
        )
        deterministic = self._deterministic(train_data)
        hmm_inputs = deterministic[["log_return", "trend_20", "garman_klass_20"]]
        self.regime_model.fit(hmm_inputs, trend_column="trend_20")
        unscaled = deterministic.join(self.regime_model.forward_probabilities(hmm_inputs))
        self.feature_order = self._candidate_columns(unscaled)
        complete = unscaled.loc[:, self.feature_order].replace([np.inf, -np.inf], np.nan).dropna()
        if complete.empty:
            raise ValueError("training data has no complete feature rows")
        self.scaler.fit(complete)
        self.dataset_range = (str(train_data.index.min()), str(train_data.index.max()))
        return self

    def transform(self, data: pd.DataFrame, history_context: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.feature_order:
            raise RuntimeError("pipeline is not fitted")
        history = history_context if history_context is not None else data.iloc[:0]
        if not history.empty and not data.empty and history.index.max() >= data.index.min():
            raise ValueError("history_context must end before data begins")
        combined = pd.concat([history, data]).sort_index()
        if combined.index.has_duplicates:
            raise ValueError("feature input contains duplicate timestamps")
        unscaled = self._unscaled(combined).loc[data.index, self.feature_order]
        valid = unscaled.replace([np.inf, -np.inf], np.nan).dropna()
        output = pd.DataFrame(index=valid.index, columns=self.feature_order, dtype=float)
        if not valid.empty:
            output.loc[:, :] = self.scaler.transform(valid)
        return output

    def manifest(self) -> dict[str, Any]:
        if self.fracdiff_selection is None or not self.feature_order:
            raise RuntimeError("pipeline is not fitted")
        return {
            "schema_version": 1,
            "feature_order": list(self.feature_order),
            "fracdiff": asdict(self.fracdiff_selection),
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "variance": self.scaler.var_.tolist(),
            },
            "hmm": self.regime_model.parameters(),
            "config_hash": self.config_hash,
            "dataset_range": list(self.dataset_range),
            "dependency_versions": _dependency_versions(),
            "git_sha": _git_sha(),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        destination.with_suffix(destination.suffix + ".manifest.json").write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CausalFeaturePipeline":
        loaded = pickle.loads(Path(path).read_bytes())
        if not isinstance(loaded, cls):
            raise TypeError("artifact is not a CausalFeaturePipeline")
        return loaded
