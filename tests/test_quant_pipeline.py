from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant import (
    ALPHA_FORECAST_COLUMNS,
    CausalAlphaEnsemble,
    CausalFeaturePipeline,
    CausalRegimeModel,
    forward_return_targets,
    rolling_wavelet_features,
)


def _bars(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.002, rows)))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.001, 0.004, rows)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1 + spread),
            "Low": np.minimum(open_, close) * (1 - spread),
            "Close": close,
            "Volume": rng.uniform(1, 10, rows),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
    )


class QuantPipelineTests(unittest.TestCase):
    def test_forward_targets_use_next_executable_m1_open(self) -> None:
        decisions = pd.date_range("2025-01-01 01:00", periods=3, freq="h", tz="UTC")
        m1_index = pd.date_range("2025-01-01 01:01", periods=180, freq="min", tz="UTC")
        prices = np.arange(100.0, 280.0)
        m1 = pd.DataFrame({"Open": prices}, index=m1_index)
        targets = forward_return_targets(decisions, m1, horizons=(1,))
        self.assertAlmostEqual(float(targets.iloc[0, 0]), np.log(160 / 100))

    def test_alpha_ensemble_is_training_fitted_and_reports_uncertainty(self) -> None:
        index = pd.date_range("2023-01-01", periods=600, freq="h", tz="UTC")
        signal = np.sin(np.arange(600) / 20)
        features = pd.DataFrame({"signal": signal, "lag": np.roll(signal, 1)}, index=index)
        targets = pd.DataFrame(
            {f"target_{horizon}h": signal * horizon / 10_000 for horizon in (4, 12, 24)},
            index=index,
        )
        ensemble = CausalAlphaEnsemble(random_state=3).fit(features.iloc[:500], targets.iloc[:500])
        forecast = ensemble.transform(features.iloc[500:], round_trip_cost=0.001)
        self.assertEqual(tuple(forecast.columns[:6]), ALPHA_FORECAST_COLUMNS)
        self.assertTrue(bool(np.isfinite(forecast.to_numpy()).all()))
        self.assertTrue(bool(forecast["alpha_target_exposure"].between(0, 1).all()))

    def test_wavelets_and_hmm_forward_filter_are_append_stable(self) -> None:
        bars = _bars(500)
        past_wavelets = rolling_wavelet_features(bars["Close"].iloc[:400])
        full_wavelets = rolling_wavelet_features(bars["Close"])
        pd.testing.assert_frame_equal(past_wavelets, full_wavelets.iloc[:400])

        features = pd.DataFrame(
            {
                "return": np.log(bars["Close"]).diff(),
                "trend": np.log(bars["Close"] / bars["Close"].ewm(span=20, adjust=False).mean()),
                "volatility": np.log(bars["High"] / bars["Low"]),
            }
        )
        model = CausalRegimeModel(random_state=3).fit(features.iloc[:300], "trend")
        past = model.forward_probabilities(features.iloc[:400])
        full = model.forward_probabilities(features)
        np.testing.assert_allclose(past, full.iloc[:400], equal_nan=True)
        np.testing.assert_allclose(full.dropna().sum(axis=1), 1.0)

    def test_pipeline_scaling_context_and_artifact_manifest(self) -> None:
        bars = _bars()
        training, first, future = bars.iloc[:600], bars.iloc[600:720], bars.iloc[720:]
        pipeline = CausalFeaturePipeline(fracdiff_threshold=1e-3, random_state=5).fit(training)
        past = pipeline.transform(first, history_context=training)
        combined = pipeline.transform(pd.concat([first, future]), history_context=training)
        pd.testing.assert_frame_equal(past, combined.loc[past.index])

        with tempfile.TemporaryDirectory() as folder:
            path = pipeline.save(Path(folder) / "features.pkl")
            manifest = json.loads(path.with_suffix(".pkl.manifest.json").read_text())
            self.assertEqual(manifest["feature_order"], list(pipeline.feature_order))
            self.assertNotIn("wavelet_a3_endpoint", pipeline.feature_order)
            self.assertIn("hmm", manifest)
            self.assertEqual(len(manifest["hmm"]["input_scale"]), 3)
            restored = CausalFeaturePipeline.load(path)
            pd.testing.assert_frame_equal(past, restored.transform(first, history_context=training))


if __name__ == "__main__":
    unittest.main()
