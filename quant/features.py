from __future__ import annotations
import numpy as np, pandas as pd

VOLATILITY_WINDOWS = (5, 15, 60, 240, 1_440, 10_080)

def true_range(frame: pd.DataFrame) -> pd.Series:
    h, l, c = frame["High"].to_numpy(float), frame["Low"].to_numpy(float), frame["Close"].to_numpy(float)
    pc = np.empty_like(c); pc[0] = np.nan; pc[1:] = c[:-1]
    return pd.Series(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), index=frame.index)

def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def _positive(frame: pd.DataFrame) -> pd.DataFrame:
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive for log-volatility estimators")
    return frame

def parkinson_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    _positive(frame)
    h, l = frame["High"].to_numpy(float), frame["Low"].to_numpy(float)
    var = pd.Series(np.log(h / l) ** 2, index=frame.index).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(var.clip(lower=0)).rename(f"parkinson_{window}")

def garman_klass_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    _positive(frame)
    h, l, c, o = frame["High"].to_numpy(float), frame["Low"].to_numpy(float), frame["Close"].to_numpy(float), frame["Open"].to_numpy(float)
    var = pd.Series(0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2, index=frame.index).rolling(window).mean()
    return np.sqrt(var.clip(lower=0)).rename(f"garman_klass_{window}")

def yang_zhang_volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    if window < 2: raise ValueError("Yang-Zhang window must be at least 2")
    _positive(frame)
    h, l, c, o = frame["High"].to_numpy(float), frame["Low"].to_numpy(float), frame["Close"].to_numpy(float), frame["Open"].to_numpy(float)
    pc = np.empty_like(c); pc[0] = np.nan; pc[1:] = c[:-1]
    overnight, open_close = pd.Series(np.log(o / pc), index=frame.index), pd.Series(np.log(c / o), index=frame.index)
    rs = pd.Series(np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c), index=frame.index)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = overnight.rolling(window).var() + k * open_close.rolling(window).var() + (1 - k) * rs.rolling(window).mean()
    return np.sqrt(var.clip(lower=0)).rename(f"yang_zhang_{window}")

def add_realized_volatility_features(m1: pd.DataFrame) -> pd.DataFrame:
    out, lr = m1.copy(), np.log(m1["Close"]).diff()
    lr_sq = lr ** 2
    for w in VOLATILITY_WINDOWS: out[f"rv_{w}m"] = np.sqrt(lr_sq.rolling(w).sum())
    for s, l in zip(VOLATILITY_WINDOWS, VOLATILITY_WINDOWS[1:]): out[f"rv_ratio_{s}m_{l}m"] = out[f"rv_{s}m"] / out[f"rv_{l}m"].replace(0, np.nan)
    for w in (60, 240, 1_440):
        r = lr.rolling(w)
        out[f"return_skew_{w}m"], out[f"return_kurt_{w}m"] = r.skew(), r.kurt()
    return out

def add_range_volatility_features(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = frame.copy()
    out[f"parkinson_{window}"] = parkinson_volatility(frame, window)
    out[f"garman_klass_{window}"] = garman_klass_volatility(frame, window)
    out[f"yang_zhang_{window}"] = yang_zhang_volatility(frame, window)
    return out

def order_book_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    tot = bid_size + ask_size
    return ((bid_size - ask_size) / tot.replace(0, np.nan)).rename("obi")

def micro_price(bid_price: pd.Series, bid_size: pd.Series, ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    tot = bid_size + ask_size
    return ((ask_price * bid_size + bid_price * ask_size) / tot.replace(0, np.nan)).rename("micro_price")
