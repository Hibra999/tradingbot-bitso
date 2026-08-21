from __future__ import annotations

import unittest

import pandas as pd

from rl.training import _attach_scaled_features


class TrainingFeatureAssemblyTests(unittest.TestCase):
    def test_scaled_features_replace_overlapping_raw_inputs(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        raw = pd.DataFrame({"Close": [100.0, 101.0], "rv_60m": [0.1, 0.2]}, index=index)
        features = pd.DataFrame({"rv_60m": [-1.0, 1.0], "trend_20": [0.5, 0.6]}, index=index)

        result = _attach_scaled_features(raw, features)

        self.assertFalse(result.columns.has_duplicates)
        self.assertEqual(result["rv_60m"].tolist(), [-1.0, 1.0])
        self.assertEqual(result["Close"].tolist(), [100.0, 101.0])


if __name__ == "__main__":
    unittest.main()
