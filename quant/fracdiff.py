from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ADFSelection:
    d: float
    p_value: float
    raw_correlation: float
    weight_count: int
    tested: tuple[tuple[float, float], ...]


@lru_cache(maxsize=256)
def _cached_weights(d: float, threshold: float, max_size: int) -> tuple[float, ...]:
    weights = [1.0]
    for lag in range(1, max_size):
        weight = -weights[-1] * (d - lag + 1) / lag
        if abs(weight) < threshold:
            break
        weights.append(weight)
    return tuple(weights)


def fracdiff_weights(d: float, threshold: float = 1e-5, max_size: int = 100_000) -> np.ndarray:
    if not 0 <= d <= 1:
        raise ValueError("d must be in [0, 1]")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if max_size < 1:
        raise ValueError("max_size must be positive")

    return np.asarray(_cached_weights(d, threshold, max_size), dtype=np.float64)


def fixed_width_fracdiff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    weights = fracdiff_weights(d, threshold=threshold, max_size=len(series))
    values = series.astype(float).to_numpy()
    result = np.convolve(values, weights, mode="full")[: len(values)]
    result[: len(weights) - 1] = np.nan
    invalid = ~np.isfinite(values)
    if invalid.any():
        cumulative = np.concatenate(([0], np.cumsum(invalid)))
        right = np.arange(1, len(values) + 1)
        contaminated = cumulative[right] > cumulative[np.maximum(right - len(weights), 0)]
        result[contaminated] = np.nan
    return pd.Series(result, index=series.index, name=f"fracdiff_{d:.2f}")


def select_adf_d(
    train_series: pd.Series,
    *,
    threshold: float = 1e-5,
    significance: float = 0.01,
    grid: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21)),
) -> ADFSelection:
    """Select the smallest stationary d using training data only."""
    clean = train_series.astype(float).replace([np.inf, -np.inf], np.nan)
    tested: list[tuple[float, float]] = []
    for d in grid:
        transformed = fixed_width_fracdiff(clean, d, threshold).dropna()
        try:
            p_value = float(adfuller(transformed, regression="c", autolag="AIC")[1]) if len(transformed) >= 20 else np.nan
        except (ValueError, np.linalg.LinAlgError):
            p_value = np.nan
        tested.append((d, p_value))
        if np.isfinite(p_value) and p_value < significance:
            aligned = pd.concat([clean.rename("raw"), transformed.rename("diff")], axis=1).dropna()
            correlation = float(aligned.corr().iloc[0, 1]) if len(aligned) > 1 else np.nan
            return ADFSelection(
                d=d,
                p_value=p_value,
                raw_correlation=correlation,
                weight_count=len(fracdiff_weights(d, threshold, max_size=len(clean))),
                tested=tuple(tested),
            )
    raise ValueError(f"No fractional-difference candidate passed ADF p < {significance}; tested={tested}")
