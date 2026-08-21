from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant import CausalFeaturePipeline, CausalRegimeModel, rolling_wavelet_features


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
            self.assertIn("hmm", manifest)
            self.assertEqual(len(manifest["hmm"]["input_scale"]), 3)
            restored = CausalFeaturePipeline.load(path)
            pd.testing.assert_frame_equal(past, restored.transform(first, history_context=training))

    def test_segmented_and_gapped_pipeline_fit(self) -> None:
        bars = _bars(800)
        seg1, seg2 = bars.iloc[:350], bars.iloc[450:]
        pipeline_seg = CausalFeaturePipeline(fracdiff_threshold=1e-3, random_state=5).fit((seg1, seg2))
        self.assertTrue(len(pipeline_seg.feature_order) > 0)
        self.assertEqual(len(pipeline_seg.dataset_range), 2)

        gapped = pd.concat([seg1, seg2])
        pipeline_gap = CausalFeaturePipeline(fracdiff_threshold=1e-3, random_state=5).fit(gapped)
        self.assertEqual(pipeline_seg.feature_order, pipeline_gap.feature_order)

    def test_hmm_forward_probabilities_outlier_resilience(self) -> None:
        bars = _bars(400)
        features = pd.DataFrame(
            {
                "return": np.log(bars["Close"]).diff(),
                "trend": np.log(bars["Close"] / bars["Close"].ewm(span=20, adjust=False).mean()),
                "volatility": np.log(bars["High"] / bars["Low"]),
            }
        )
        model = CausalRegimeModel(random_state=3).fit(features.iloc[:200], "trend")
        corrupted = features.copy()
        corrupted.iloc[250, 0] = 50.0  # 5000% return outlier
        corrupted.iloc[250, 1] = 50.0
        probs = model.forward_probabilities(corrupted)
        self.assertFalse(probs.iloc[251:].isnull().all().any())
        valid = probs.dropna()
        np.testing.assert_allclose(valid.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
