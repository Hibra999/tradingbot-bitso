from __future__ import annotations

import unittest

import numpy as np

from validation import (
    centered_block_bootstrap_test,
    institutional_metrics,
    moving_block_monte_carlo,
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

    def test_monte_carlo_is_batched_seeded_and_reports_ruin(self) -> None:
        returns = np.array([0.01, -0.005, 0.002, -0.001] * 20)
        first = moving_block_monte_carlo(returns, paths=100, horizon=40, block_size=4, batch_size=13, seed=9)
        second = moving_block_monte_carlo(returns, paths=100, horizon=40, block_size=4, batch_size=17, seed=9)
        np.testing.assert_allclose(first.equity_cone, second.equity_cone)
        self.assertEqual(first.manifest(), second.manifest())
        self.assertTrue(0 <= first.ruin_probability_20 <= 1)
        self.assertEqual(list(first.equity_cone.columns), ["p05", "p25", "p50", "p75", "p95"])

    def test_extreme_loss_bounds_and_empty_edge_cases(self) -> None:
        from validation.metrics import sharpe_ratio, sortino_ratio, calmar_ratio, drawdowns

        catastrophic_returns = [0.01, -0.02, -1.0, -1.5, 0.05]
        s = sharpe_ratio(catastrophic_returns)
        self.assertTrue(np.isfinite(s))
        dd = drawdowns(catastrophic_returns)
        self.assertAlmostEqual(float(dd.min()), -1.0)

        # Single element or empty
        self.assertTrue(np.isnan(sharpe_ratio([0.05])))
        self.assertTrue(np.isnan(sharpe_ratio([])))
        self.assertTrue(np.isnan(sortino_ratio([0.05])))
        self.assertTrue(np.isnan(calmar_ratio([0.05])))

        # Constant returns
        self.assertTrue(np.isnan(sharpe_ratio([0.01, 0.01, 0.01])))


if __name__ == "__main__":
    unittest.main()
