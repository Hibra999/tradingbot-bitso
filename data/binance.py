from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .storage import BAR_SCHEMA_VERSION, bar_metadata, read_parquet, read_parquet_metadata, write_parquet

_SPOT_URL = "https://api.binance.com/api/v3/klines"
_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_HOUR_MS = 60 * 60 * 1_000


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _utc(value: pd.Timestamp | str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tz is None else result.tz_convert("UTC")


def _validate_hourly(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{source} bars must be non-empty, chronological, and unique")
    gaps = frame.index.to_series().diff().dropna()
    if bool((gaps != pd.Timedelta(hours=1)).any()):
        raise ValueError(f"{source} bars contain gaps; market context fails closed")
    if bool((frame[["Open", "High", "Low", "Close"]] <= 0).any().any()):
        raise ValueError(f"{source} bars contain non-positive prices")
    return frame


def _download_klines(
    symbol: str,
    market: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    url, limit = (_SPOT_URL, 1_000) if market == "spot" else (_FUTURES_URL, 1_500)
    current = int(start.timestamp() * 1_000)
    stop = int(end.timestamp() * 1_000)
    rows: list[list[object]] = []
    with httpx.Client(timeout=30.0) as client:
        while current < stop:
            response = client.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": "1h",
                    "startTime": current,
                    "endTime": stop - 1,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise RuntimeError("Binance kline response is not a list")
            if not page:
                break
            rows.extend(page)
            next_start = int(page[-1][0]) + _HOUR_MS
            if next_start <= current:
                raise RuntimeError("Binance kline pagination did not advance")
            current = next_start
    if not rows:
        raise RuntimeError(f"Binance returned no {market} bars for {symbol}")
    frame = pd.DataFrame(
        {
            "Open": [float(row[1]) for row in rows],
            "High": [float(row[2]) for row in rows],
            "Low": [float(row[3]) for row in rows],
            "Close": [float(row[4]) for row in rows],
            "Volume": [float(row[5]) for row in rows],
            "QuoteVolume": [float(row[7]) for row in rows],
            "TradeCount": [float(row[8]) for row in rows],
            "TakerBuyVolume": [float(row[9]) for row in rows],
        },
        index=pd.to_datetime([int(row[0]) + _HOUR_MS for row in rows], unit="ms", utc=True),
    )
    frame.index.name = "Time"
    return _validate_hourly(frame.loc[~frame.index.duplicated(keep="last")].sort_index(), f"Binance {market}")


def load_binance_klines(
    symbol: str,
    market: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    cache_dir: str | Path,
    cache_only: bool,
) -> pd.DataFrame:
    if market not in {"spot", "futures"}:
        raise ValueError("Binance market must be spot or futures")
    start_time, end_time = _utc(start).floor("h"), _utc(end).ceil("h")
    path = Path(cache_dir) / f"binance_{market}_{symbol}_1h.parquet"
    cached: pd.DataFrame | None = None
    if path.exists():
        metadata = read_parquet_metadata(path)
        if metadata.get("bar_schema_version") != BAR_SCHEMA_VERSION or metadata.get("bar_timestamp") != "close":
            raise ValueError(f"Binance cache has an unsupported bar schema: {path}")
        cached = _validate_hourly(read_parquet(path).sort_index(), f"cached Binance {market}")
        if metadata.get("content_sha256") != _frame_hash(cached):
            raise ValueError(f"Binance cache checksum failed: {path}")
    covered = (
        cached is not None
        and cached.index.min() <= start_time + pd.Timedelta(hours=1)
        and cached.index.max() >= end_time
    )
    if cache_only and not covered:
        raise RuntimeError(
            f"Binance {market} context cache does not cover {symbol}; run once with --no-cache-only"
        )
    if not covered:
        fetch_start = (
            max(start_time, cached.index.max() - pd.Timedelta(hours=1))
            if cached is not None
            else start_time
        )
        downloaded = _download_klines(symbol, market, fetch_start, end_time)
        cached = (
            pd.concat((cached, downloaded)).loc[lambda value: ~value.index.duplicated(keep="last")].sort_index()
            if cached is not None
            else downloaded
        )
        cached = _validate_hourly(cached, f"merged Binance {market}")
        metadata = bar_metadata(source=f"binance_{market}", interval="1h")
        metadata["content_sha256"] = _frame_hash(cached)
        write_parquet(cached, path, metadata)
    result = cached.loc[(cached.index > start_time) & (cached.index <= end_time)]
    return _validate_hourly(result, f"sliced Binance {market}")


def _download_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    current, stop = int(start.timestamp() * 1_000), int(end.timestamp() * 1_000)
    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=30.0) as client:
        while current < stop:
            response = client.get(
                _FUNDING_URL,
                params={"symbol": symbol, "startTime": current, "endTime": stop, "limit": 1_000},
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise RuntimeError("Binance funding response is not a list")
            if not page:
                break
            rows.extend(page)
            next_start = int(page[-1]["fundingTime"]) + 1
            if next_start <= current:
                raise RuntimeError("Binance funding pagination did not advance")
            current = next_start
    if not rows:
        raise RuntimeError(f"Binance returned no funding history for {symbol}")
    frame = pd.DataFrame(
        {"FundingRate": [float(row["fundingRate"]) for row in rows]},
        index=pd.to_datetime([int(row["fundingTime"]) for row in rows], unit="ms", utc=True),
    )
    frame.index.name = "Time"
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def load_binance_funding(
    symbol: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    cache_dir: str | Path,
    cache_only: bool,
) -> pd.DataFrame:
    start_time, end_time = _utc(start), _utc(end)
    path = Path(cache_dir) / f"binance_funding_{symbol}.parquet"
    cached = read_parquet(path).sort_index() if path.exists() else None
    if cached is not None:
        metadata = read_parquet_metadata(path)
        if metadata.get("schema_version") != 1 or metadata.get("source") != "binance_funding":
            raise ValueError(f"Binance funding cache has an unsupported schema: {path}")
        if metadata.get("content_sha256") != _frame_hash(cached):
            raise ValueError(f"Binance funding cache checksum failed: {path}")
    covered = cached is not None and cached.index.min() <= start_time and cached.index.max() >= end_time - pd.Timedelta(hours=8)
    if cache_only and not covered:
        raise RuntimeError(f"Binance funding cache does not cover {symbol}; run once with --no-cache-only")
    if not covered:
        fetch_start = cached.index.max() + pd.Timedelta(milliseconds=1) if cached is not None else start_time
        downloaded = _download_funding(symbol, fetch_start, end_time)
        cached = (
            pd.concat((cached, downloaded)).loc[lambda value: ~value.index.duplicated(keep="last")].sort_index()
            if cached is not None
            else downloaded
        )
        write_parquet(
            cached,
            path,
            {
                "source": "binance_funding",
                "schema_version": 1,
                "content_sha256": _frame_hash(cached),
            },
        )
    return cached.loc[(cached.index >= start_time) & (cached.index <= end_time)]


def load_binance_context(
    project_symbol: str,
    decision_index: pd.DatetimeIndex,
    *,
    cache_dir: str | Path,
    cache_only: bool,
) -> pd.DataFrame:
    if project_symbol not in {"BTC/USD", "ETH/USD"} or decision_index.empty:
        raise ValueError("Binance context requires BTC/USD or ETH/USD decisions")
    own = "BTCUSDT" if project_symbol == "BTC/USD" else "ETHUSDT"
    other = "ETHUSDT" if own == "BTCUSDT" else "BTCUSDT"
    start, end = decision_index.min() - pd.Timedelta(hours=200), decision_index.max()
    spot = load_binance_klines(own, "spot", start, end, cache_dir=cache_dir, cache_only=cache_only)
    cross = load_binance_klines(other, "spot", start, end, cache_dir=cache_dir, cache_only=cache_only)
    futures = load_binance_klines(own, "futures", start, end, cache_dir=cache_dir, cache_only=cache_only)
    funding = load_binance_funding(own, start, end, cache_dir=cache_dir, cache_only=cache_only)
    base = spot.reindex(decision_index)
    cross_close = cross["Close"].reindex(decision_index)
    futures_close = futures["Close"].reindex(decision_index)
    if bool(base[["Close", "Volume", "TradeCount", "TakerBuyVolume"]].isna().any().any()):
        raise ValueError("Binance spot context does not align exactly with decision timestamps")
    if bool(cross_close.isna().any()) or bool(futures_close.isna().any()):
        raise ValueError("Binance cross-asset or futures context does not align with decisions")
    own_return = np.log(base["Close"]).diff()
    other_return = np.log(cross_close).diff()
    output = pd.DataFrame(index=decision_index)
    output["ctx_quote_volume_24h"] = np.log1p(base["QuoteVolume"]) - np.log1p(base["QuoteVolume"]).rolling(24).mean()
    output["ctx_trade_count_24h"] = np.log1p(base["TradeCount"]) - np.log1p(base["TradeCount"]).rolling(24).mean()
    output["ctx_taker_buy_imbalance"] = 2 * base["TakerBuyVolume"] / base["Volume"].replace(0, np.nan) - 1
    output["ctx_futures_basis"] = np.log(futures_close / base["Close"])
    aligned_funding = funding["FundingRate"].reindex(decision_index, method="ffill")
    if bool(aligned_funding.isna().any()):
        raise ValueError("Binance funding context is unavailable at a decision timestamp")
    output["ctx_funding_rate"] = aligned_funding
    for horizon in (4, 12, 24):
        output[f"ctx_relative_strength_{horizon}h"] = (
            np.log(base["Close"] / base["Close"].shift(horizon))
            - np.log(cross_close / cross_close.shift(horizon))
        )
    output["ctx_cross_correlation_168h"] = own_return.rolling(168).corr(other_return)
    covariance = own_return.rolling(168).cov(other_return)
    output["ctx_cross_beta_168h"] = covariance / other_return.rolling(168).var().replace(0, np.nan)
    return output.replace([np.inf, -np.inf], np.nan)
