from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
