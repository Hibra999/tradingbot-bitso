from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rl import BracketExecutionCore, ppo_intent, qrdqn_action_table, sac_intent


class EnvironmentTests(unittest.TestCase):
    def test_entry_is_next_m1_tick_and_same_bar_tie_is_stop_first(self) -> None:
        decisions = pd.DataFrame(
            {"Open": [100, 100], "High": [101, 101], "Low": [99, 99], "Close": [100, 100], "atr": [1, 1]},
            index=pd.date_range("2025-01-01 01:00", periods=2, freq="h", tz="UTC"),
        )
        m1_index = pd.date_range("2025-01-01 01:01", periods=60, freq="min", tz="UTC")
        m1 = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0}, index=m1_index)
        m1.iloc[0] = [100, 102, 98, 100]
        core = BracketExecutionCore(decisions, m1, commission_rate=0, holding_cost_r=0)
        intent = ppo_intent([1, 0, 0], model_id="m", book="btc_usd", timestamp=decisions.index[0].to_pydatetime())
        result = core.execute_interval(0, intent)
        self.assertEqual(core.trades[0]["entry_time"], m1_index[0])
        self.assertGreater(core.trades[0]["entry_time"], decisions.index[0])
        self.assertEqual(core.trades[0]["reason"], "stop")
        self.assertAlmostEqual(result.realized_r, -1.0)

    def test_action_mappings_are_bounded(self) -> None:
        self.assertEqual(len(qrdqn_action_table()), 129)
        timestamp = pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime()
        self.assertEqual(sac_intent([0.09, 0.01, 1.5, 2], model_id="m", book="btc_usd", timestamp=timestamp).direction, 0)
        self.assertEqual(sac_intent([-0.5, 0.01, 1.5, 2], model_id="m", book="btc_usd", timestamp=timestamp).direction, -1)


if __name__ == "__main__":
    unittest.main()
