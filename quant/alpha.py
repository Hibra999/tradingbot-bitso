from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ALPHA_HORIZONS = (4, 12, 24)
ALPHA_FORECAST_COLUMNS = tuple(
    column
    for horizon in ALPHA_HORIZONS
    for column in (f"alpha_mean_{horizon}h", f"alpha_dispersion_{horizon}h")
)
ALPHA_TARGET_COLUMN = "alpha_target_exposure"


def forward_return_targets(
    decision_index: pd.DatetimeIndex,
    m1_bars: pd.DataFrame,
    horizons: tuple[int, ...] = ALPHA_HORIZONS,
) -> pd.DataFrame:
    if not decision_index.is_monotonic_increasing or decision_index.has_duplicates:
        raise ValueError("decision index must be chronological and unique")
    if not m1_bars.index.is_monotonic_increasing or m1_bars.index.has_duplicates:
        raise ValueError("M1 bars must be chronological and unique")
    open_times = m1_bars.index - pd.Timedelta(minutes=1)
    prices = m1_bars["Open"].to_numpy(dtype=float, copy=False)
    entry = open_times.searchsorted(decision_index, side="left")
    output: dict[str, np.ndarray] = {}
    for horizon in horizons:
        exit_positions = open_times.searchsorted(
            decision_index + pd.Timedelta(hours=horizon), side="left"
        )
        valid = (entry < len(prices)) & (exit_positions < len(prices))
        values = np.full(len(decision_index), np.nan, dtype=float)
        values[valid] = np.log(prices[exit_positions[valid]] / prices[entry[valid]])
        output[f"target_{horizon}h"] = values
    return pd.DataFrame(output, index=decision_index)


@dataclass(frozen=True)
class _AlphaExpert:
    horizon: int
    window_months: int
    model_name: str
    validation_mse: float
    baseline_mse: float
    permutation_mse: float
    weight: float
    model: Any


class CausalAlphaEnsemble:
    def __init__(self, *, random_state: int = 0):
        self.random_state = int(random_state)
        self.feature_order: tuple[str, ...] = ()
        self.experts: dict[int, tuple[_AlphaExpert, ...]] = {}
        self.training_range: tuple[str, str] = ()

    @staticmethod
    def _model(name: str, random_state: int):
        if name == "ridge":
            return Ridge(alpha=10.0)
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        )

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> "CausalAlphaEnsemble":
        if features.empty or not features.index.is_monotonic_increasing or features.index.has_duplicates:
            raise ValueError("alpha training features must be non-empty, chronological, and unique")
        aligned = features.join(targets, how="inner").replace([np.inf, -np.inf], np.nan)
        self.feature_order = tuple(features.columns)
        self.training_range = (str(features.index.min()), str(features.index.max()))
        for horizon in ALPHA_HORIZONS:
            target_column = f"target_{horizon}h"
            clean = aligned.dropna(subset=[*self.feature_order, target_column])
            if len(clean) < 200:
                raise ValueError(f"alpha horizon {horizon}h requires at least 200 complete rows")
            candidates: list[tuple[int, str, float, float, float, Any]] = []
            for window_months in (12, 36):
                start = clean.index.max() - pd.DateOffset(months=window_months)
                window = clean.loc[clean.index >= start]
                split = max(100, int(len(window) * 0.90))
                train_end = max(split - max(ALPHA_HORIZONS), 1)
                if train_end < 100 or len(window) - split < 20:
                    continue
                x_train = window.iloc[:train_end].loc[:, self.feature_order]
                y_train = window.iloc[:train_end][target_column]
                x_validation = window.iloc[split:].loc[:, self.feature_order]
                y_validation = window.iloc[split:][target_column]
                baseline_mse = float(
                    np.mean(np.square(y_validation.to_numpy() - float(y_train.mean())))
                )
                rng = np.random.default_rng(self.random_state + horizon + window_months)
                permutation = Ridge(alpha=10.0).fit(
                    x_train, rng.permutation(y_train.to_numpy())
                )
                permutation_error = np.asarray(permutation.predict(x_validation)) - y_validation.to_numpy()
                permutation_mse = float(np.mean(np.square(permutation_error)))
                for model_name in ("ridge", "hist_gradient_boosting"):
                    validator = self._model(model_name, self.random_state)
                    validator.fit(x_train, y_train)
                    error = np.asarray(validator.predict(x_validation)) - y_validation.to_numpy()
                    mse = float(np.mean(np.square(error)))
                    fitted = self._model(model_name, self.random_state)
                    fitted.fit(window.loc[:, self.feature_order], window[target_column])
                    candidates.append(
                        (
                            window_months,
                            model_name,
                            max(mse, 1e-12),
                            max(baseline_mse, 1e-12),
                            max(permutation_mse, 1e-12),
                            fitted,
                        )
                    )
            if len(candidates) < 2:
                raise ValueError(f"alpha horizon {horizon}h has insufficient expert windows")
            inverse = np.asarray([1 / item[2] for item in candidates], dtype=float)
            weights = inverse / inverse.sum()
            self.experts[horizon] = tuple(
                _AlphaExpert(
                    horizon,
                    window,
                    name,
                    mse,
                    baseline_mse,
                    permutation_mse,
                    float(weight),
                    model,
                )
                for (
                    window,
                    name,
                    mse,
                    baseline_mse,
                    permutation_mse,
                    model,
                ), weight in zip(candidates, weights)
            )
        return self

    def transform(self, features: pd.DataFrame, *, round_trip_cost: float) -> pd.DataFrame:
        if not self.feature_order or set(self.experts) != set(ALPHA_HORIZONS):
            raise RuntimeError("alpha ensemble is not fitted")
        if round_trip_cost < 0:
            raise ValueError("round-trip cost must be non-negative")
        values = features.loc[:, self.feature_order].replace([np.inf, -np.inf], np.nan)
        complete = values.dropna()
        output = pd.DataFrame(index=features.index, columns=ALPHA_FORECAST_COLUMNS, dtype=float)
        for horizon, experts in self.experts.items():
            if complete.empty:
                continue
            predictions = np.column_stack([expert.model.predict(complete) for expert in experts])
            weights = np.asarray([expert.weight for expert in experts])
            mean = predictions @ weights
            dispersion = np.sqrt(np.maximum(((predictions - mean[:, None]) ** 2) @ weights, 0.0))
            output.loc[complete.index, f"alpha_mean_{horizon}h"] = mean
            output.loc[complete.index, f"alpha_dispersion_{horizon}h"] = dispersion
        edge = output["alpha_mean_12h"] - 1.28 * output["alpha_dispersion_12h"] - round_trip_cost
        scale = np.maximum(2 * output["alpha_dispersion_12h"].to_numpy(dtype=float), 0.001)
        raw_target = np.clip(edge.to_numpy(dtype=float) / scale, 0.0, 1.0)
        target = np.zeros(len(output), dtype=float)
        for index, value in enumerate(np.nan_to_num(raw_target, nan=0.0)):
            previous = target[index - 1] if index else 0.0
            target[index] = previous if abs(value - previous) < 0.10 else value
        output[ALPHA_TARGET_COLUMN] = target
        return output

    def manifest(self) -> dict[str, Any]:
        if not self.feature_order:
            raise RuntimeError("alpha ensemble is not fitted")
        diagnostic_pass = all(
            any(
                expert.validation_mse < expert.baseline_mse
                and expert.validation_mse < expert.permutation_mse
                for expert in experts
            )
            for experts in self.experts.values()
        )
        return {
            "schema_version": 1,
            "feature_order": list(self.feature_order),
            "forecast_columns": list(ALPHA_FORECAST_COLUMNS),
            "primary_horizon_hours": 12,
            "diagnostic_pass": diagnostic_pass,
            "training_range": list(self.training_range),
            "experts": {
                str(horizon): [
                    {
                        "window_months": expert.window_months,
                        "model": expert.model_name,
                        "validation_mse": expert.validation_mse,
                        "validation_skill": 1 - expert.validation_mse / expert.baseline_mse,
                        "permutation_mse": expert.permutation_mse,
                        "weight": expert.weight,
                    }
                    for expert in experts
                ]
                for horizon, experts in self.experts.items()
            },
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
    def load(cls, path: str | Path) -> "CausalAlphaEnsemble":
        loaded = pickle.loads(Path(path).read_bytes())
        if not isinstance(loaded, cls):
            raise TypeError("artifact is not a CausalAlphaEnsemble")
        return loaded
