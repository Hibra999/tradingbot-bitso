from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def rolling_wavelet_features(
    series: pd.Series,
    *,
    window: int = 256,
    wavelet: str = "db4",
    level: int = 3,
) -> pd.DataFrame:
    """Compute each row from its trailing window only."""
    if window < 2**level:
        raise ValueError("wavelet window is too short for the requested level")
    values = series.astype(float).to_numpy(copy=True)
    columns = [f"wavelet_a{level}_endpoint", *(f"wavelet_d{i}_energy" for i in range(level, 0, -1))]
    output = np.full((len(values), len(columns)), np.nan, dtype=float)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        coefficients = pywt.wavedec(windows, wavelet, level=level, mode="periodization", axis=-1)
        output[window - 1 :, 0] = coefficients[0][:, -1]
        output[window - 1 :, 1:] = np.column_stack(
            tuple(np.mean(np.square(detail), axis=1) for detail in coefficients[1:])
        )
    return pd.DataFrame(output, index=series.index, columns=columns)
