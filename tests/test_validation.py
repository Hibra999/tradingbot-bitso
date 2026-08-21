from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data import make_sliding_folds
from validation import CPCVSplitter, DomainRandomizer, SeedHarness


class ValidationTests(unittest.TestCase):
    def test_cpcv_purges_overlap_embargo_and_splits_episodes(self) -> None:
        index = pd.date_range("2025-01-01", periods=60, freq="h", tz="UTC")
        ends = index + pd.Timedelta(hours=2)
        splitter = CPCVSplitter(temporal_groups=6, test_groups=2, embargo_bars=2, max_holding_bars=3)
        folds = splitter.split(index, ends)
        self.assertEqual(len(folds), 15)
        for fold in folds:
            for train_index in fold.train_indices:
                for test_index in fold.test_indices:
                    self.assertFalse(index[train_index] <= ends[test_index] and ends[train_index] >= index[test_index])
            for segment in fold.episode_segments:
                self.assertTrue(bool((np.diff(segment) == 1).all()))
            for group in fold.test_groups:
                group_end = (group + 1) * 10 - 1
                self.assertFalse(set(range(group_end + 1, min(group_end + 3, len(index)))) & set(fold.train_indices))

    def test_randomization_and_seed_manifest_are_reproducible(self) -> None:
        features = pd.DataFrame(np.zeros((20, 3)), columns=list("abc"))
        atr = pd.Series(np.ones(20))
        first = DomainRandomizer(42).perturb(features, atr, 0.5)
        second = DomainRandomizer(42).perturb(features, atr, 0.5)
        np.testing.assert_allclose(first.features, second.features)
        np.testing.assert_allclose(first.spread, second.spread)
        self.assertTrue(bool(((first.latency_ticks >= 1) & (first.latency_ticks <= 3)).all()))

        evaluation = SeedHarness(tuple(range(10))).run(
            lambda seed: {"sharpe": 1.1 + seed / 100, "dsr_z": 0.2, "return": float(seed)}
        )
        self.assertTrue(evaluation.sri_pass)
        self.assertEqual(evaluation.manifest()["seeds"], list(range(10)))
        self.assertIn("iqr", evaluation.aggregate["return"])
        self.assertIn("iqm", evaluation.aggregate["return"])
        self.assertIn("ci95_low", evaluation.aggregate["return"])

    def test_sliding_folds_have_non_overlapping_chronological_evaluations(self) -> None:
        index = pd.date_range("2018-01-01", "2025-01-01", freq="h", inclusive="left", tz="UTC")
        frame = pd.DataFrame({"Close": np.arange(len(index), dtype=float) + 1}, index=index)
        folds = make_sliding_folds(
            frame,
            train_years=3,
            val_months=6,
            test_months=6,
            step_months=6,
            embargo_bars=24,
        )
        self.assertGreaterEqual(len(folds), 2)
        previous_end = None
        for training, validation, evaluation in folds:
            self.assertLess(training.index.max(), validation.index.min())
            self.assertLess(validation.index.max(), evaluation.index.min())
            if previous_end is not None:
                self.assertLess(previous_end, evaluation.index.min())
            previous_end = evaluation.index.max()


if __name__ == "__main__":
    unittest.main()
