from __future__ import annotations

import unittest

import numpy as np

from validation import (
    centered_block_bootstrap_test,
    institutional_metrics,
    moving_block_monte_carlo,
    probability_of_backtest_overfitting,
)


class MetricsTests(unittest.TestCase):
    def test_dsr_semantics_and_seeded_hypothesis_tests(self) -> None:
        rng = np.random.default_rng(12)
        returns = rng.normal(0.001, 0.01, 500)
        metrics = institutional_metrics(
            returns,
            trial_sharpes=[0.1, 0.2, 0.3, 0.4],
            periods_per_year=365,
            bootstrap_repetitions=200,
            seed=4,
        )
        self.assertAlmostEqual(metrics["dsr_p_value"], 1 - metrics["dsr_probability"])
        self.assertLess(metrics["cvar_99"], float("inf"))
        first = centered_block_bootstrap_test(returns, repetitions=100, seed=8)
        second = centered_block_bootstrap_test(returns, repetitions=100, seed=8)
        self.assertEqual(first, second)
        self.assertLess(first["ci95_low"], first["ci95_high"])

    def test_monte_carlo_is_batched_seeded_and_reports_ruin(self) -> None:
        returns = np.array([0.01, -0.005, 0.002, -0.001] * 20)
        first = moving_block_monte_carlo(returns, paths=100, horizon=40, block_size=4, batch_size=13, seed=9)
        second = moving_block_monte_carlo(returns, paths=100, horizon=40, block_size=4, batch_size=17, seed=9)
        np.testing.assert_allclose(first.equity_cone, second.equity_cone)
        self.assertEqual(first.manifest(), second.manifest())
        self.assertTrue(0 <= first.ruin_probability_20 <= 1)
        self.assertEqual(list(first.equity_cone.columns), ["p05", "p25", "p50", "p75", "p95"])

    def test_pbo_detects_selection_rank_reversal(self) -> None:
        selection = [[2.0, 1.0], [1.0, 2.0]]
        reversed_evaluation = [[0.0, 1.0], [1.0, 0.0]]
        aligned_evaluation = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(
            probability_of_backtest_overfitting(selection, reversed_evaluation)["pbo_probability"],
            1.0,
        )
        self.assertEqual(
            probability_of_backtest_overfitting(selection, aligned_evaluation)["pbo_probability"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
