from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import bar_metadata, load_binance_context, read_parquet, resample_ohlcv, write_parquet
from execution import TradeIntent
from quant import (
    ALPHA_FORECAST_COLUMNS,
    ALPHA_TARGET_COLUMN,
    CausalAlphaEnsemble,
    HAR_RV_COLUMNS,
    CausalFeaturePipeline,
    add_realized_volatility_features,
    align_m1_features_to_decisions,
    atr,
    fracdiff_weights,
)

from .actions import target_exposure_intent
from .candidates import (
    PUFFER_ABSOLUTE_ACTION_ENCODING,
    PUFFER_ACTION_ENCODING,
    PUFFER_ALGORITHM,
    PUFFER_LEGACY_ACTION_ENCODING,
    PufferPolicyRunner,
    load_puffer_policy,
)


@dataclass(frozen=True)
class PolicyDecision:
    intent: TradeIntent
    atr: Decimal
    decision_time: pd.Timestamp


class LivePolicyRuntime:
    def __init__(
        self,
        manifest: dict[str, Any],
        market_data_dir: str | Path,
        *,
        minimum_shadow_days: int | None = None,
    ):
        bundle = manifest.get("artifact_bundle")
        if not isinstance(bundle, dict) or bundle.get("action_contract") != "target_exposure_long_cash_v1":
            raise PermissionError("live policy requires a target-exposure artifact bundle")
        self.model_id = str(manifest["model_id"])
        self.algorithm = str(bundle["algorithm"])
        self.policy_action_encoding = str(
            bundle.get("policy_action_encoding", PUFFER_LEGACY_ACTION_ENCODING)
        )
        self.book = str(bundle["book"])
        self.feature_z_limit = float(bundle.get("feature_z_limit", 10.0))
        if self.feature_z_limit <= 0:
            raise PermissionError("feature drift limit must be positive")
        self.pipeline = CausalFeaturePipeline.load(bundle["feature_pipeline_path"])
        self.alpha = CausalAlphaEnsemble.load(bundle["alpha_pipeline_path"])
        if self.policy_action_encoding == PUFFER_ACTION_ENCODING:
            expected_order = (
                self.pipeline.feature_order
                + ALPHA_FORECAST_COLUMNS
                + (ALPHA_TARGET_COLUMN,)
            )
        elif self.policy_action_encoding in {
            PUFFER_ABSOLUTE_ACTION_ENCODING,
            PUFFER_LEGACY_ACTION_ENCODING,
        }:
            expected_order = self.pipeline.feature_order + ALPHA_FORECAST_COLUMNS
        else:
            raise PermissionError("live policy action encoding is unsupported")
        if tuple(bundle.get("feature_order", ())) != expected_order:
            raise PermissionError("manifest, feature, and alpha artifact orders do not match")
        if self.alpha.feature_order != self.pipeline.feature_order:
            raise PermissionError("alpha artifact expects a different base feature order")
        self.feature_order = expected_order
        self.round_trip_cost = (
            2 * float(bundle.get("commission_bps", 10.0))
            + float(bundle.get("base_spread_bps", 2.0))
        ) / 10_000
        self.max_risk_fraction = float(bundle.get("max_risk_fraction", 0.005))
        if not 0 < self.max_risk_fraction <= 0.03:
            raise PermissionError("target-exposure risk fraction is invalid")
        self.market_context = str(bundle.get("market_context", ""))
        if self.market_context not in {"none", "binance_public_v1"}:
            raise PermissionError("live policy market context contract is unsupported")
        if self.pipeline.fracdiff_selection is None:
            raise PermissionError("live feature pipeline must be fitted")
        self.required_h1_bars = max(
            self.pipeline.wavelet_window,
            len(fracdiff_weights(self.pipeline.fracdiff_selection.d, self.pipeline.fracdiff_threshold)),
            168,
        )
        self.minimum_shadow_days = int(
            bundle.get("minimum_shadow_days", 90)
            if minimum_shadow_days is None
            else minimum_shadow_days
        )
        if self.minimum_shadow_days < 0:
            raise PermissionError("minimum shadow period cannot be negative")
        self.required_m1_bars = max(
            self.required_h1_bars * 60,
            self.minimum_shadow_days * 24 * 60,
        )
        self.max_m1_bars = self.required_m1_bars + 60
        self.model = self._load_model(bundle["model_path"])
        self._validate_action_space()
        self.history_path = Path(market_data_dir) / f"{self.book}_m1.parquet"
        self.m1 = self._load_history()

    def _validate_action_space(self) -> None:
        action_space = self.model.action_space
        if tuple(action_space.shape) != (1,):
            raise PermissionError("continuous-control artifact action space is invalid")

    def _load_model(self, path: str):
        if self.algorithm != PUFFER_ALGORITHM:
            raise ValueError(f"unsupported live algorithm: {self.algorithm}")
        policy = load_puffer_policy(
            path,
            len(self.feature_order) + 7,
            action_encoding=self.policy_action_encoding,
        )
        return PufferPolicyRunner(policy)

    def _load_history(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"],
                index=pd.DatetimeIndex([], tz="UTC"),
                dtype=float,
            )
        history = read_parquet(self.history_path).sort_index()
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not isinstance(history.index, pd.DatetimeIndex) or any(column not in history for column in required):
            raise ValueError("persisted Bitso M1 history is invalid")
        return history.loc[~history.index.duplicated(keep="last"), required].tail(self.max_m1_bars)

    def _decision_frame(self) -> pd.DataFrame:
        decision = resample_ohlcv(self.m1, "1h")
        decision["atr"] = atr(decision)
        realized = add_realized_volatility_features(self.m1, include_moments=False)
        decision = decision.join(
            align_m1_features_to_decisions(realized[list(HAR_RV_COLUMNS)], decision.index)
        )
        if self.market_context == "binance_public_v1":
            project_symbol = "BTC/USD" if self.book == "btc_usd" else "ETH/USD"
            decision = decision.join(
                load_binance_context(
                    project_symbol,
                    decision.index,
                    cache_dir=self.history_path.parent / f"{self.book}_context",
                    cache_only=False,
                )
            )
        return decision

    def _action_intent(self, action: Any, timestamp: pd.Timestamp) -> TradeIntent:
        return target_exposure_intent(
            action,
            model_id=self.model_id,
            book=self.book,
            timestamp=timestamp.to_pydatetime(),
            max_risk_fraction=self.max_risk_fraction,
        )

    def on_closed_m1(
        self,
        timestamp: pd.Timestamp,
        candle: dict[str, float],
        *,
        position_direction: int = 0,
        position_age_bars: int = 0,
        equity_return: float = 0.0,
        position_entry_price: float = 0.0,
        equity_drawdown: float = 0.0,
        target_exposure: float = 0.0,
        unrealized_return: float | None = None,
    ) -> PolicyDecision | None:
        if timestamp.tz is None:
            raise ValueError("live M1 timestamps must be timezone-aware")
        row = pd.DataFrame(
            {
                "Open": [float(candle["open"])],
                "High": [float(candle["high"])],
                "Low": [float(candle["low"])],
                "Close": [float(candle["close"])],
                "Volume": [float(candle.get("volume", 0.0))],
            },
            index=pd.DatetimeIndex([timestamp]),
        )
        self.m1 = pd.concat((self.m1, row)).sort_index()
        self.m1 = self.m1.loc[~self.m1.index.duplicated(keep="last")].tail(self.max_m1_bars)
        if timestamp.minute != 0:
            return None
        write_parquet(
            self.m1,
            self.history_path,
            bar_metadata(source="bitso_public_book_midprice", interval="1min"),
        )
        required_m1 = self.required_m1_bars
        if len(self.m1) < required_m1:
            return None
        recent = self.m1.index[-required_m1:]
        if not bool((recent.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()):
            return None
        decision = self._decision_frame()
        if len(decision) < 2 or decision.index[-1] != timestamp:
            return None
        current = decision.iloc[[-1]]
        features = self.pipeline.transform(current, history_context=decision.iloc[:-1])
        if current.index[-1] not in features.index:
            return None
        forecasts = self.alpha.transform(features, round_trip_cost=self.round_trip_cost)
        combined = features.join(forecasts)
        feature_values = combined.loc[current.index[-1], self.feature_order].to_numpy(dtype=np.float32)
        if not bool(np.isfinite(feature_values).all()) or float(np.max(np.abs(feature_values))) > self.feature_z_limit:
            raise RuntimeError("live feature drift exceeded the approved artifact limit")
        close = float(current["Close"].iloc[0])
        atr_value = max(float(current["atr"].iloc[0]), 1e-12)
        entry_distance = (
            (close - position_entry_price) * position_direction / atr_value
            if position_direction and position_entry_price > 0
            else 0.0
        )
        volatility = atr_value / max(close, 1e-12)
        observed_unrealized = equity_return if unrealized_return is None else unrealized_return
        state = np.asarray(
            [
                target_exposure,
                entry_distance,
                observed_unrealized,
                position_age_bars / 24,
                equity_drawdown,
                volatility,
                self.round_trip_cost,
            ],
            dtype=np.float32,
        )
        if not 0 <= target_exposure <= 1 or not bool(np.isfinite(state).all()):
            raise RuntimeError("live risk state is outside the approved observation contract")
        observation = np.concatenate((feature_values, state))
        action = self.model.predict(observation, deterministic=True)
        return PolicyDecision(
            self._action_intent(action, timestamp),
            Decimal(str(float(current["atr"].iloc[0]))),
            timestamp,
        )
