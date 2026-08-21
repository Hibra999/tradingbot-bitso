from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller

@dataclass(frozen=True)
class ADFSelection:
    d: float
    p_value: float
    raw_correlation: float
    weight_count: int
    tested: tuple[tuple[float, float], ...]

def fracdiff_weights(d: float, threshold: float = 1e-5, max_size: int = 100_000) -> np.ndarray:
    if not 0 <= d <= 1: raise ValueError("d must be in [0, 1]")
    if not 0 < threshold < 1: raise ValueError("threshold must be in (0, 1)")
    if max_size < 1: raise ValueError("max_size must be positive")
    w = [1.0]
    for k in range(1, max_size):
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < threshold: break
        w.append(nxt)
    return np.asarray(w, dtype=np.float64)

def fixed_width_fracdiff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    w = fracdiff_weights(d, threshold=threshold, max_size=len(series))
    v = series.astype(float).to_numpy()
    res = np.convolve(v, w, mode="full")[: len(v)]
    res[: len(w) - 1] = np.nan
    inv = ~np.isfinite(v)
    if inv.any():
        cnt = np.convolve(inv.astype(int), np.ones(len(w), dtype=int), mode="full")[: len(v)]
        res[cnt > 0] = np.nan
    return pd.Series(res, index=series.index, name=f"fracdiff_{d:.2f}")

def select_adf_d(
    train_series: pd.Series,
    *,
    threshold: float = 1e-5,
    significance: float = 0.01,
    grid: tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21)),
) -> ADFSelection:
    clean = train_series.astype(float).replace([np.inf, -np.inf], np.nan)
    tested: list[tuple[float, float]] = []
    for d in grid:
        transformed = fixed_width_fracdiff(clean, d, threshold).dropna()
        try: p_value = float(adfuller(transformed, regression="c", autolag="AIC")[1]) if len(transformed) >= 20 else np.nan
        except (ValueError, np.linalg.LinAlgError): p_value = np.nan
        tested.append((d, p_value))
        if np.isfinite(p_value) and p_value < significance:
            al = pd.concat([clean.rename("raw"), transformed.rename("diff")], axis=1).dropna().to_numpy()
            corr = float(np.corrcoef(al[:, 0], al[:, 1])[0, 1]) if len(al) > 1 else np.nan
            return ADFSelection(d=d, p_value=p_value, raw_correlation=corr, weight_count=len(fracdiff_weights(d, threshold, max_size=len(clean))), tested=tuple(tested))
    raise ValueError(f"No fractional-difference candidate passed ADF p < {significance}; tested={tested}")
