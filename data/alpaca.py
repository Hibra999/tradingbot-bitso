from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# pandas resample rule -> Alpaca TimeFrame.
_PANDAS_RULE_TO_ALPACA: dict[str, tuple[int, TimeFrameUnit]] = {
    "1min": (1, TimeFrameUnit.Minute),
    "5min": (5, TimeFrameUnit.Minute),
    "15min": (15, TimeFrameUnit.Minute),
    "30min": (30, TimeFrameUnit.Minute),
    "1h": (1, TimeFrameUnit.Hour),
    "4h": (4, TimeFrameUnit.Hour),
    "1D": (1, TimeFrameUnit.Day),
}

# Common Alpaca crypto pairs (ticker without the "/").
_CRYPTO_SYMBOLS = {
    "BTCUSD", "ETHUSD", "USDTUSD", "USDCUSD", "SOLUSD",
    "ADAUSD", "DOTUSD", "AVAXUSD", "LTCUSD", "BCHUSD",
    "LINKUSD", "UNIUSD", "XRPUSD", "DOGEUSD", "SHIBUSD",
    "MATICUSD", "AAVEUSD", "MKRUSD", "YFIUSD", "EOSUSD",
    "CRVUSD", "GRTUSD", "BALUSD", "SNXUSD", "SUSHIUSD",
}

# Alpaca returns 15-min delayed IEX data on the free plan; fine for research.
_DEFAULT_PAGE_SIZE = 10_000  # SDK paginates internally; 10k bars per request
_DEFAULT_START = "2020-01-01"  # default history depth when start_date not given


def _get_credentials() -> tuple[str, str]:
    key = (os.getenv("ALPACA_API_KEY") or "").strip()
    secret = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
    if not key or not secret or "pon_tu" in key.lower():
        raise RuntimeError(
            "Alpaca credentials missing. Open '.env' (created from .env.example) "
            "and set ALPACA_API_KEY / ALPACA_SECRET_KEY with your keys from "
            "https://alpaca.markets (Paper Trading -> API Keys)."
        )
    return key, secret


def _is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper().replace("/", "")
    return "/" in symbol.upper() or s in _CRYPTO_SYMBOLS


def _pandas_rule_to_timeframe(pandas_rule: str) -> TimeFrame:
    if pandas_rule not in _PANDAS_RULE_TO_ALPACA:
        raise ValueError(
            f"Alpaca does not support timeframe '{pandas_rule}'. "
            f"Supported: {', '.join(_PANDAS_RULE_TO_ALPACA)}"
        )
    amount, unit = _PANDAS_RULE_TO_ALPACA[pandas_rule]
    return TimeFrame(amount, unit)


def _bars_to_frame(response) -> pd.DataFrame:
    """Flatten the SDK response into a single-symbol OHLCV DataFrame.

    Handles both wrappers: a BarSet/CryptoBarSet (has a ``.df`` property via
    TimeSeriesMixin) and the raw dict form {symbol: [bar_dict, ...]}.
    """
    if hasattr(response, "df"):
        df = response.df
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        elif isinstance(df.index, pd.MultiIndex):
            df.index = df.index.droplevel(0)
        return df

    # RawData fallback: {symbol: [{"t": ..., "o": ..., "h": ..., "l": ...,
    #                              "c": ..., "v": ..., "n": ..., "vw": ...}]}
    data = getattr(response, "data", response) or {}
    frames = []
    for symbol, bars in data.items():
        if isinstance(bars, pd.DataFrame):
            frames.append(bars)
        else:
            frames.append(pd.DataFrame(bars))
    if not frames:
        raise RuntimeError("Alpaca response contained no bar data.")
    df = pd.concat(frames, ignore_index=True)
    for time_col in ("t", "timestamp"):
        if time_col in df.columns:
            df.index = pd.DatetimeIndex(df[time_col], name="Time")
            df = df.drop(columns=[time_col])
            break
    return df


def load_alpaca_ohlcv(
    symbol: str = "GLD",
    pandas_rule: str = "1min",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_days_for_demo: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = "data/cache",
    timestamp_is_bar_open: Optional[bool] = None,
    feed: str = "iex",
    cache_only: bool = False,
) -> pd.DataFrame:
    """Download OHLCV bars from Alpaca Markets and return a standardized,
    timezone-aware (UTC) DataFrame indexed by bar CLOSE time with columns
    Open/High/Low/Close/Volume — the same contract as load_mt_ohlcv_csv().

    Parameters
    ----------
    symbol : str
        Ticker (US stock/ETF, e.g. GLD) or crypto pair (e.g. BTC/USD).
    pandas_rule : str
        Bar size as a pandas resample rule: "1min", "5min", "15min", "1h", "4h", "1D".
    start_date, end_date : optional str
        YYYY-MM-DD range (UTC). Defaults: start 2020-01-01, end = now.
    max_days_for_demo : optional int
        Keep only the last N days of data (same semantics as config.max_days_for_demo).
    cache_dir : optional path
        If given, the first download is stored as
        <cache_dir>/alpaca_<SYMBOL>_<rule>.parquet and reused on later runs: if it
        already covers the requested start, only the missing tail is
        re-downloaded (delta refresh) and merged — repeated runs cost almost
        no API quota.
timestamp_is_bar_open : optional bool
        True → Alpaca's bar-OPEN timestamps are shifted forward by one bar so
        the index is bar-close time (project convention). Default: True for
        stocks (Alpaca labels those by open), False for crypto (labeled by close).
    feed : str
        Stock data feed: "iex" (free plan, 15-min delayed) or "sip" (paid,
        real-time consolidated tape). Ignored for crypto.
    cache_only : bool
        If True, return only cached data without making any API calls.
        Raises RuntimeError if no cache file exists for this symbol/timeframe.

    Returns
    -------
    pd.DataFrame
        Indexed by close time (UTC, name "Time"); Open/High/Low/Close/Volume.
    """
    symbol = symbol.upper()
    is_crypto = _is_crypto_symbol(symbol)
    api_symbol = f"{symbol[:3]}/{symbol[3:]}" if is_crypto and "/" not in symbol else symbol

    if timestamp_is_bar_open is None:
        timestamp_is_bar_open = not is_crypto

    tf = _pandas_rule_to_timeframe(pandas_rule)
    bar_delta = pd.to_timedelta(pandas_rule)
    if pandas_rule == "1D":
        bar_delta = pd.Timedelta(days=1)

    end = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(start_date, tz="UTC") if start_date else pd.Timestamp(_DEFAULT_START, tz="UTC")
    if max_days_for_demo is not None:
        start = max(start, end - pd.Timedelta(days=max_days_for_demo))

    # ── Local cache −────────────────────────────────────────────────────────
    # Reuses whatever already-downloaded bars we have so long as the requested
    # start is covered.  Two cases:
    #   • cache covers [start, end)            -> pure cache hit, zero API calls
    #   • cache covers start but not the end  -> delta refresh: download ONLY
    #     the missing tail and merge with the cache instead of re-downloading
    #     the whole range.
    # A cache older than the *start* of the request is useless (API history
    # cannot go backwards), so it is replaced by a full download.
    cache_path: Optional[Path] = None
    cached: Optional[pd.DataFrame] = None
    fetch_start: Optional[pd.Timestamp] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"alpaca_{symbol.replace('/', '_')}_{pandas_rule}.parquet"
        if cache_path.exists():
            cached = _read_cached_parquet(cache_path)

    # -- cache_only mode: return whatever is cached, no API calls --------
    if cache_only:
        if cached is not None and len(cached):
            print(f"Alpaca: cache-only -> {cache_path.name if cache_path else 'memory'} ({len(cached):,} bars)")
            return _slice_range(cached, start, min(end, cached.index.max()), max_days_for_demo)
        raise RuntimeError(
            f"Cache-only mode but no cached data found for {symbol} ({pandas_rule}). "
            f"Run once with CACHE_ONLY=false to download data first."
        )

    if cached is not None and len(cached):
        end_tolerance = pd.Timedelta(days=7) if pandas_rule == "1D" else pd.Timedelta(days=1)
        if cached.index.max() >= end - end_tolerance:
            print(f"Alpaca: cache hit -> {cache_path.name} ({len(cached):,} bars)")
            return _slice_range(cached, start, min(end, cached.index.max()), max_days_for_demo)
        print(f"Alpaca: cache covers {cached.index.min()} -> {cached.index.max()}; refreshing the tail only")
        fetch_start = cached.index.max() - bar_delta  # re-fetch the boundary bar for a clean merge
    else:
        cached = None

    api_key, secret_key = _get_credentials()

    # SDK paginates automatically over the whole range as long as no
    # request-level ``limit`` caps it (a limit of 10_000 would stop after the
    # first page), so a multi-year M1 range only costs a few sequential calls.
    if fetch_start is None:
        fetch_start = start - bar_delta - pd.Timedelta(minutes=30)  # small overlap for the close-time index
    if is_crypto:
        client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        request = CryptoBarsRequest(
            symbol_or_symbols=api_symbol,
            timeframe=tf,
            start=fetch_start.to_pydatetime(),
            end=end.to_pydatetime(),
        )
        print(f"Alpaca: downloading {symbol} {pandas_rule} bars {fetch_start} -> {end} (crypto) ...")
        response = client.get_crypto_bars(request)
    else:
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=api_symbol,
            timeframe=tf,
            start=fetch_start.to_pydatetime(),
            end=end.to_pydatetime(),
            adjustment="all",  # splits & dividends
            feed=feed,  # "iex" (free/15-min delay) | "sip" (paid/real-time)
        )
        print(f"Alpaca: downloading {symbol} {pandas_rule} bars {fetch_start} -> {end} (stock) ...")
        response = client.get_stock_bars(request)

    raw = _bars_to_frame(response)
    if raw.empty:
        if cached is not None and len(cached):
            print(f"Alpaca: no new bars since {cached.index.max()} (market closed?); reusing cached data.")
            return _slice_range(cached, start, end, max_days_for_demo)
        raise RuntimeError(
            f"Alpaca returned no bars for {api_symbol} ({pandas_rule}, "
            f"{fetch_start} -> {end}). Check the symbol and the range."
        )

    df = raw.rename(columns={
        # BarSet.df (current SDK) uses full lowercase names; the raw dict path
        # and older builds use single-letter codes. Accept both.
        "open": "Open", "o": "Open",
        "high": "High", "h": "High",
        "low": "Low", "l": "Low",
        "close": "Close", "c": "Close",
        "volume": "Volume", "v": "Volume",
    })
    df.index = pd.DatetimeIndex(df.index, name="Time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[OHLCV].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    if timestamp_is_bar_open:
        df.index = df.index + bar_delta

    # Merge with the existing cache (tail refresh) and persist the union so
    # the next run only downloads whatever happened after this one.
    if cached is not None and len(cached):
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    print(f"Alpaca: cache total {len(df):,} bars ({df.index.min()} -> {df.index.max()}). Saving -> {cache_path}")
    if cache_path is not None:
        _write_cached_parquet(df, cache_path)

    return _slice_range(df, start, end, max_days_for_demo)


def _slice_range(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_days_for_demo: Optional[int] = None,
) -> pd.DataFrame:
    df = df.loc[start:end]
    if max_days_for_demo is not None and len(df):
        df = df.loc[max(df.index.max() - pd.Timedelta(days=max_days_for_demo), start):]
    return df


def _read_cached_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df.loc[~idx.isna()].copy()
        df.index = pd.DatetimeIndex(idx[~idx.isna()], name="Time")
        return df[OHLCV].sort_index()
    except Exception:
        return None


def _write_cached_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd")
