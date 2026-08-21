from __future__ import annotations

import numpy as np
import pandas as pd

VOLATILITY_WINDOWS = (5, 15, 60, 240, 1_440, 10_080)
_LOG2 = np.log(2.0)


def true_range(frame: pd.DataFrame) -> pd.Series:
    high, low, close = (frame[column].to_numpy(dtype=float, copy=False) for column in ("High", "Low", "Close"))
    if not len(close):
        return pd.Series(index=frame.index, dtype=float)
    previous = np.empty_like(close)
    previous[0], previous[1:] = np.nan, close[:-1]
    values = np.fmax(high - low, np.fmax(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(values, index=frame.index)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _positive(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close"]
    if (frame[columns] <= 0).any().any():
        raise ValueError("OHLC prices must be positive for log-volatility estimators")
    return frame


def parkinson_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    _positive(frame)
    variance = np.log(frame["High"] / frame["Low"]).pow(2).rolling(window).mean() / (4 * _LOG2)
    return np.sqrt(variance.clip(lower=0)).rename(f"parkinson_{window}")


def garman_klass_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    _positive(frame)
    log_hl = np.log(frame["High"] / frame["Low"])
    log_co = np.log(frame["Close"] / frame["Open"])
    variance = (0.5 * log_hl.pow(2) - (2 * _LOG2 - 1) * log_co.pow(2)).rolling(window).mean()
    return np.sqrt(variance.clip(lower=0)).rename(f"garman_klass_{window}")


def yang_zhang_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    if window < 2:
        raise ValueError("Yang-Zhang window must be at least 2")
    _positive(frame)
    overnight = np.log(frame["Open"] / frame["Close"].shift())
    open_close = np.log(frame["Close"] / frame["Open"])
    rogers_satchell = (
        np.log(frame["High"] / frame["Open"]) * np.log(frame["High"] / frame["Close"])
        + np.log(frame["Low"] / frame["Open"]) * np.log(frame["Low"] / frame["Close"])
    )
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = overnight.rolling(window).var() + k * open_close.rolling(window).var() + (1 - k) * rogers_satchell.rolling(window).mean()
    return np.sqrt(variance.clip(lower=0)).rename(f"yang_zhang_{window}")


def add_realized_volatility_features(m1: pd.DataFrame) -> pd.DataFrame:
    """Add a causal realized-volatility term matrix to close-labelled M1 bars."""
    out = m1.copy()
    log_returns = np.log(out["Close"]).diff()
    squared = log_returns.pow(2)
    for window in VOLATILITY_WINDOWS:
        out[f"rv_{window}m"] = np.sqrt(squared.rolling(window).sum())
    for short, long in zip(VOLATILITY_WINDOWS, VOLATILITY_WINDOWS[1:]):
        out[f"rv_ratio_{short}m_{long}m"] = out[f"rv_{short}m"] / out[f"rv_{long}m"].replace(0, np.nan)
    for window in (60, 240, 1_440):
        out[f"return_skew_{window}m"] = log_returns.rolling(window).skew()
        out[f"return_kurt_{window}m"] = log_returns.rolling(window).kurt()
    return out


def add_range_volatility_features(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = frame.copy()
    out[f"parkinson_{window}"] = parkinson_volatility(frame, window)
    out[f"garman_klass_{window}"] = garman_klass_volatility(frame, window)
    out[f"yang_zhang_{window}"] = yang_zhang_volatility(frame, window)
    return out


def order_book_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    total = bid_size + ask_size
    return ((bid_size - ask_size) / total.replace(0, np.nan)).rename("obi")


def micro_price(
    bid_price: pd.Series,
    bid_size: pd.Series,
    ask_price: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    total = bid_size + ask_size
    return ((ask_price * bid_size + bid_price * ask_size) / total.replace(0, np.nan)).rename("micro_price")
