from __future__ import annotations
import numpy as np, pandas as pd, pywt

def rolling_wavelet_features(series: pd.Series, *, window: int = 256, wavelet: str = "db4", level: int = 3) -> pd.DataFrame:
    if window < 2**level: raise ValueError("wavelet window is too short for the requested level")
    values = series.astype(float).to_numpy(copy=True)
    columns = [f"wavelet_a{level}_endpoint", *(f"wavelet_d{i}_energy" for i in range(level, 0, -1))]
    N, C = len(values), len(columns)
    output = np.full((N, C), np.nan, dtype=float)
    for end in range(window - 1, N):
        coeffs = pywt.wavedec(values[end - window + 1 : end + 1], wavelet, level=level, mode="periodization")
        output[end, 0] = coeffs[0][-1]
        for idx, d in enumerate(coeffs[1:], 1): output[end, idx] = float(np.mean(d ** 2))
    return pd.DataFrame(output, index=series.index, columns=columns)
