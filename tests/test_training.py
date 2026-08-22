from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rl.training import _attach_scaled_features, _full_walk_forward_windows, _smoke_research_fold


class TrainingFeatureAssemblyTests(unittest.TestCase):
    def test_smoke_uses_one_purged_chronological_fold_with_enough_training_rows(self) -> None:
        index = pd.date_range("2025-01-01", periods=1_440, freq="h", tz="UTC")
        positions = np.minimum(np.arange(len(index)) + 24, len(index) - 1)
        interval_end = pd.DatetimeIndex(index[positions])

        fold = _smoke_research_fold(
            index,
            interval_end,
            embargo_bars=200,
            max_holding_bars=24,
        )

        self.assertGreater(len(fold.train_indices), 500)
        self.assertLess(fold.train_indices.max(), fold.validation_indices.min())
        self.assertLess(fold.validation_indices.max(), fold.episode_segments[0].min())
        self.assertLess(interval_end[fold.train_indices].max(), index[fold.validation_indices].min())

    def test_scaled_features_replace_overlapping_raw_inputs(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        raw = pd.DataFrame({"Close": [100.0, 101.0], "rv_60m": [0.1, 0.2]}, index=index)
        features = pd.DataFrame({"rv_60m": [-1.0, 1.0], "trend_20": [0.5, 0.6]}, index=index)

        result = _attach_scaled_features(raw, features)

        self.assertFalse(result.columns.has_duplicates)
        self.assertEqual(result["rv_60m"].tolist(), [-1.0, 1.0])
        self.assertEqual(result["Close"].tolist(), [100.0, 101.0])

    def test_full_windows_adapt_to_minute_history_shorter_than_five_years(self) -> None:
        index = pd.date_range("2022-01-01", "2025-04-02", freq="h", inclusive="left", tz="UTC")
        frame = pd.DataFrame({"Close": np.ones(len(index))}, index=index)

        windows, effective_train_months = _full_walk_forward_windows(
            frame,
            train_months=36,
            validation_months=6,
            evaluation_months=6,
            step_months=6,
            embargo_bars=24,
        )

        self.assertGreaterEqual(len(windows), 2)
        self.assertGreaterEqual(effective_train_months, 12)
        self.assertLess(effective_train_months, 36)
        self.assertLess(windows[0][2].index.max(), windows[1][2].index.min())


if __name__ == "__main__":
    unittest.main()
