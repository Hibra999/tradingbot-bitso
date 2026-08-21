from __future__ import annotations
import json, pickle, subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from config import AppConfig
from .features import add_range_volatility_features, atr
from .fracdiff import ADFSelection, fixed_width_fracdiff, select_adf_d
from .regimes import CausalRegimeModel
from .wavelets import rolling_wavelet_features

_DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "statsmodels", "hmmlearn", "PyWavelets")

def _dependency_versions() -> dict[str, str]:
    res: dict[str, str] = {}
    for dep in _DEPENDENCIES:
        try: res[dep] = version(dep)
        except PackageNotFoundError: res[dep] = "unavailable"
    return res

def _git_sha() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 else "unknown"

class CausalFeaturePipeline:
    def __init__(self, *, config_hash: str | None = None, fracdiff_threshold: float = 1e-5, wavelet_window: int = 256, random_state: int = 0):
        self.config_hash = config_hash or AppConfig().config_hash
        self.fracdiff_threshold = fracdiff_threshold
        self.wavelet_window = wavelet_window
        self.regime_model = CausalRegimeModel(random_state=random_state)
        self.scaler = StandardScaler()
        self.fracdiff_selection: ADFSelection | None = None
        self.feature_order: tuple[str, ...] = ()
        self.dataset_range: tuple[str, str] = ()

    def _deterministic(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fracdiff_selection is None: raise RuntimeError("FracDiff parameter is not fitted")
        out = add_range_volatility_features(frame, 20)
        log_c = np.log(out["Close"])
        out["log_return"] = log_c.diff()
        out["trend_20"] = np.log(out["Close"] / out["Close"].ewm(span=20, adjust=False).mean())
        out["atr_close"] = atr(out, 14) / out["Close"]
        out["fracdiff_close"] = fixed_width_fracdiff(log_c, self.fracdiff_selection.d, self.fracdiff_threshold)
        return out.join(rolling_wavelet_features(log_c, window=self.wavelet_window, wavelet="db4", level=3))

    @staticmethod
    def _candidate_columns(frame: pd.DataFrame) -> tuple[str, ...]:
        core = ("log_return", "trend_20", "atr_close", "parkinson_20", "garman_klass_20", "yang_zhang_20", "fracdiff_close", "wavelet_a3_endpoint", "wavelet_d3_energy", "wavelet_d2_energy", "wavelet_d1_energy", "regime_bear", "regime_neutral", "regime_bull")
        live_ex = {"obi", "micro_price"}
        realized = tuple(c for c in frame.columns if (c.startswith("rv_") or c.startswith("return_")) and c not in live_ex)
        return core + realized

    def _unscaled(self, frame: pd.DataFrame) -> pd.DataFrame:
        det = self._deterministic(frame)
        return det.join(self.regime_model.forward_probabilities(det[["log_return", "trend_20", "garman_klass_20"]]))

    def fit(self, train_data: pd.DataFrame | tuple[pd.DataFrame, ...] | list[pd.DataFrame]) -> "CausalFeaturePipeline":
        if isinstance(train_data, pd.DataFrame):
            if train_data.empty: raise ValueError("training data must be non-empty")
            if not train_data.index.is_monotonic_increasing: raise ValueError("training data must be chronological")
            if isinstance(train_data.index, pd.DatetimeIndex) and len(train_data) > 1:
                diffs = train_data.index.to_series().diff().iloc[1:]
                median_step = diffs.median()
                if pd.notna(median_step) and median_step > pd.Timedelta(0):
                    gap_mask = diffs > (median_step * 3)
                    if gap_mask.any():
                        gap_indices = np.flatnonzero(gap_mask.to_numpy()) + 1
                        segments = [train_data.iloc[part] for part in np.split(np.arange(len(train_data)), gap_indices) if len(part) > 0]
                    else:
                        segments = [train_data]
                else:
                    segments = [train_data]
            else:
                segments = [train_data]
        elif isinstance(train_data, (tuple, list)):
            segments = [s for s in train_data if isinstance(s, pd.DataFrame) and not s.empty]
            if not segments: raise ValueError("training data has no valid non-empty segments")
            for s in segments:
                if not s.index.is_monotonic_increasing: raise ValueError("training data segments must be chronological")
        else:
            raise TypeError("training data must be a DataFrame or sequence of DataFrames")

        longest_segment = max(segments, key=len)
        self.fracdiff_selection = select_adf_d(np.log(longest_segment["Close"]), threshold=self.fracdiff_threshold)
        det_segments = [self._deterministic(seg) for seg in segments]
        hmm_in_list = [d[["log_return", "trend_20", "garman_klass_20"]].dropna() for d in det_segments]
        all_hmm_in = pd.concat(hmm_in_list)
        self.regime_model.fit(all_hmm_in, trend_column="trend_20")

        unscaled_segments = []
        for det_seg in det_segments:
            hmm_in_seg = det_seg[["log_return", "trend_20", "garman_klass_20"]]
            unscaled_segments.append(det_seg.join(self.regime_model.forward_probabilities(hmm_in_seg)))

        self.feature_order = self._candidate_columns(unscaled_segments[0])
        complete_list = [u.loc[:, self.feature_order].replace([np.inf, -np.inf], np.nan).dropna() for u in unscaled_segments]
        complete = pd.concat(complete_list)
        if complete.empty: raise ValueError("training data has no complete feature rows")
        self.scaler.fit(complete)
        min_ts = min(seg.index.min() for seg in segments)
        max_ts = max(seg.index.max() for seg in segments)
        self.dataset_range = (str(min_ts), str(max_ts))
        return self

    def transform(self, data: pd.DataFrame, history_context: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.feature_order: raise RuntimeError("pipeline is not fitted")
        history = history_context if history_context is not None else data.iloc[:0]
        if not history.empty and not data.empty and history.index.max() >= data.index.min(): raise ValueError("history_context must end before data begins")
        combined = pd.concat([history, data]).sort_index()
        if combined.index.has_duplicates: raise ValueError("feature input contains duplicate timestamps")
        unscaled = self._unscaled(combined).loc[data.index, self.feature_order]
        valid = unscaled.replace([np.inf, -np.inf], np.nan).dropna()
        output = pd.DataFrame(index=valid.index, columns=self.feature_order, dtype=float)
        if not valid.empty: output.loc[:, :] = self.scaler.transform(valid)
        return output

    def manifest(self) -> dict[str, Any]:
        if self.fracdiff_selection is None or not self.feature_order: raise RuntimeError("pipeline is not fitted")
        return {
            "schema_version": 1,
            "feature_order": list(self.feature_order),
            "fracdiff": asdict(self.fracdiff_selection),
            "scaler": {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(), "variance": self.scaler.var_.tolist()},
            "hmm": self.regime_model.parameters(),
            "config_hash": self.config_hash,
            "dataset_range": list(self.dataset_range),
            "dependency_versions": _dependency_versions(),
            "git_sha": _git_sha(),
        }

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        dest.with_suffix(dest.suffix + ".manifest.json").write_text(json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "CausalFeaturePipeline":
        loaded = pickle.loads(Path(path).read_bytes())
        if not isinstance(loaded, cls): raise TypeError("artifact is not a CausalFeaturePipeline")
        return loaded
