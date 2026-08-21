from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import read_parquet, resample_ohlcv, write_parquet
from execution import TradeIntent
from quant import (
    HAR_RV_COLUMNS,
    CausalFeaturePipeline,
    add_realized_volatility_features,
    align_m1_features_to_decisions,
    atr,
    fracdiff_weights,
)

from .actions import ppo_intent, qrdqn_intent, sac_intent
from .candidates import CVaRQRDQN, RecurrentPolicyRunner


@dataclass(frozen=True)
class PolicyDecision:
    intent: TradeIntent
    atr: Decimal
    decision_time: pd.Timestamp


class LivePolicyRuntime:
    def __init__(self, manifest: dict[str, Any], market_data_dir: str | Path):
        bundle = manifest.get("artifact_bundle")
        if not isinstance(bundle, dict) or bundle.get("action_contract") != "long_flat_spot":
            raise PermissionError("live policy requires a complete long/flat artifact bundle")
        self.model_id = str(manifest["model_id"])
        self.algorithm = str(bundle["algorithm"])
        self.book = str(bundle["book"])
        self.feature_z_limit = float(bundle.get("feature_z_limit", 10.0))
        if self.feature_z_limit <= 0:
            raise PermissionError("feature drift limit must be positive")
        self.pipeline = CausalFeaturePipeline.load(bundle["feature_pipeline_path"])
        if tuple(bundle.get("feature_order", ())) != self.pipeline.feature_order:
            raise PermissionError("manifest and feature artifact orders do not match")
        if self.pipeline.fracdiff_selection is None:
            raise PermissionError("live feature pipeline must be fitted")
        self.required_h1_bars = max(
            self.pipeline.wavelet_window,
            len(fracdiff_weights(self.pipeline.fracdiff_selection.d, self.pipeline.fracdiff_threshold)),
            168,
        )
        self.max_m1_bars = self.required_h1_bars * 60 + 60
        self.model = self._load_model(bundle["model_path"])
        self._validate_action_space()
        self.recurrent = RecurrentPolicyRunner(self.model) if self.algorithm == "recurrent_ppo" else None
        self.history_path = Path(market_data_dir) / f"{self.book}_m1.parquet"
        self.m1 = self._load_history()

    def _validate_action_space(self) -> None:
        action_space = self.model.action_space
        if self.algorithm == "recurrent_ppo":
            if tuple(int(value) for value in action_space.nvec) != (2, 4, 4):
                raise PermissionError("PPO artifact does not use the approved long/flat action space")
        elif self.algorithm == "cvar_qrdqn":
            if int(action_space.n) != 65:
                raise PermissionError("QR-DQN artifact does not use the approved long/flat action space")
        elif tuple(action_space.shape) != (4,):
            raise PermissionError("continuous-control artifact action space is invalid")

    def _load_model(self, path: str):
        if self.algorithm == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO

            return RecurrentPPO.load(path, device="auto")
        if self.algorithm == "sac":
            from stable_baselines3 import SAC

            return SAC.load(path, device="auto")
        if self.algorithm == "tqc":
            from sb3_contrib import TQC

            return TQC.load(path, device="auto")
        if self.algorithm == "cvar_qrdqn":
            return CVaRQRDQN.load(path, device="auto")
        raise ValueError(f"unsupported live algorithm: {self.algorithm}")

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
        return decision.join(
            align_m1_features_to_decisions(realized[list(HAR_RV_COLUMNS)], decision.index)
        )

    def _action_intent(self, action: Any, timestamp: pd.Timestamp) -> TradeIntent:
        common = {
            "model_id": self.model_id,
            "book": self.book,
            "timestamp": timestamp.to_pydatetime(),
            "allow_short": False,
        }
        if self.algorithm == "recurrent_ppo":
            return ppo_intent(action, **common)
        if self.algorithm in {"sac", "tqc"}:
            return sac_intent(action, **common)
        return qrdqn_intent(int(np.asarray(action).item()), **common)

    def on_closed_m1(
        self,
        timestamp: pd.Timestamp,
        candle: dict[str, float],
        *,
        position_direction: int = 0,
        position_age_bars: int = 0,
        equity_return: float = 0.0,
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
        write_parquet(self.m1, self.history_path, {"source": "bitso_public_book_midprice"})
        required_m1 = self.required_h1_bars * 60
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
        feature_values = features.iloc[-1].to_numpy(dtype=np.float32)
        if not bool(np.isfinite(feature_values).all()) or float(np.max(np.abs(feature_values))) > self.feature_z_limit:
            raise RuntimeError("live feature drift exceeded the approved artifact limit")
        observation = np.concatenate(
            (
                feature_values,
                np.asarray(
                    [position_direction, position_age_bars / 24, equity_return],
                    dtype=np.float32,
                ),
            )
        )
        action = (
            self.recurrent.predict(observation, deterministic=True)
            if self.recurrent
            else self.model.predict(observation, deterministic=True)[0]
        )
        return PolicyDecision(
            self._action_intent(action, timestamp),
            Decimal(str(float(current["atr"].iloc[0]))),
            timestamp,
        )
