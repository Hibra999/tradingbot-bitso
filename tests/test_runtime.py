from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant import ALPHA_FORECAST_COLUMNS
from rl import LivePolicyRuntime


class _Pipeline:
    feature_order = ("feature",)
    fracdiff_selection = SimpleNamespace(d=0.0)
    fracdiff_threshold = 1e-5
    wavelet_window = 1

    def transform(self, current, history_context=None):
        return pd.DataFrame({"feature": [0.5]}, index=current.index)


class _Model:
    action_space = SimpleNamespace(shape=(1,))

    def __init__(self):
        self.last_observation = None

    def predict(self, observation, deterministic=True):
        self.last_observation = observation
        return np.asarray([0.0], dtype=np.float32), None


class _Alpha:
    feature_order = ("feature",)

    def transform(self, features, *, round_trip_cost):
        return pd.DataFrame(
            {column: [0.0] for column in (*ALPHA_FORECAST_COLUMNS, "alpha_target_exposure")},
            index=features.index,
        )


class RuntimeTests(unittest.TestCase):
    def test_live_runtime_uses_closed_h1_and_long_flat_action_contract(self) -> None:
        manifest = {
            "model_id": "btc-model",
            "artifact_bundle": {
                "action_contract": "target_exposure_long_cash_v1",
                "market_context": "binance_public_v1",
                "algorithm": "sac",
                "book": "btc_usd",
                "model_path": "model.zip",
                "feature_pipeline_path": "features.pkl",
                "alpha_pipeline_path": "alpha.pkl",
                "feature_order": ["feature", *ALPHA_FORECAST_COLUMNS],
            },
        }
        decision_index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        decision = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.0, 101.0],
                "Volume": [0.0, 0.0],
                "atr": [1.0, 1.0],
            },
            index=decision_index,
        )
        with tempfile.TemporaryDirectory() as folder, patch(
            "rl.runtime.CausalFeaturePipeline.load", return_value=_Pipeline()
        ), patch(
            "rl.runtime.CausalAlphaEnsemble.load", return_value=_Alpha()
        ), patch.object(LivePolicyRuntime, "_load_model", return_value=_Model()), patch(
            "rl.runtime.write_parquet"
        ), patch.object(LivePolicyRuntime, "_decision_frame", return_value=decision):
            runtime = LivePolicyRuntime(manifest, Path(folder))
            runtime.required_m1_bars = 0
            result = runtime.on_closed_m1(
                decision_index[-1],
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
                position_direction=1,
                position_entry_price=100.0,
                target_exposure=0.4,
                unrealized_return=0.02,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.intent.direction, 0)
        self.assertEqual(result.decision_time, decision_index[-1])
        self.assertAlmostEqual(float(runtime.model.last_observation[-7]), 0.4)
        self.assertAlmostEqual(float(runtime.model.last_observation[-5]), 0.02)


if __name__ == "__main__":
    unittest.main()
