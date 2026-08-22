from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rl import (
    BracketExecutionCore,
    BracketTradingEnvV2,
    PufferTradingEnv,
    ppo_intent,
    qrdqn_action_table,
    sac_intent,
)
from validation import PerturbationConfig


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
        self.assertFalse(hasattr(core, "_m1_open_time"))
        intent = ppo_intent([1, 0, 0], model_id="m", book="btc_usd", timestamp=decisions.index[0].to_pydatetime())
        result = core.execute_interval(0, intent)
        self.assertEqual(core.trades[0]["entry_time"], decisions.index[0])
        self.assertGreaterEqual(core.trades[0]["entry_time"], decisions.index[0])
        self.assertEqual(core.trades[0]["reason"], "stop")
        self.assertAlmostEqual(result.realized_r, -1.0)

    def test_action_mappings_are_bounded(self) -> None:
        self.assertEqual(len(qrdqn_action_table()), 129)
        long_only = qrdqn_action_table(allow_short=False)
        self.assertEqual(len(long_only), 65)
        self.assertNotIn(-1, {action[0] for action in long_only})
        timestamp = pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime()
        self.assertEqual(sac_intent([0.09, 0.01, 1.5, 2], model_id="m", book="btc_usd", timestamp=timestamp).direction, 0)
        self.assertEqual(sac_intent([-0.5, 0.01, 1.5, 2], model_id="m", book="btc_usd", timestamp=timestamp).direction, -1)
        self.assertEqual(
            sac_intent(
                [-0.5, 0.01, 1.5, 2],
                model_id="m",
                book="btc_usd",
                timestamp=timestamp,
                allow_short=False,
            ).direction,
            0,
        )

    def test_open_position_is_liquidated_at_segment_end(self) -> None:
        decisions = pd.DataFrame(
            {"Open": [100, 101], "High": [101, 102], "Low": [99, 100], "Close": [100, 101], "atr": [1, 1]},
            index=pd.date_range("2025-01-01 01:00", periods=2, freq="h", tz="UTC"),
        )
        m1_index = pd.date_range("2025-01-01 01:01", periods=60, freq="min", tz="UTC")
        m1 = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 100.0, "Close": 101.0}, index=m1_index)
        core = BracketExecutionCore(decisions, m1, commission_rate=0, holding_cost_r=0)
        intent = ppo_intent([1, 0, 3], model_id="m", book="btc_usd", timestamp=decisions.index[0].to_pydatetime())
        result = core.execute_interval(0, intent)
        self.assertEqual(core.position.direction, 0)
        self.assertEqual(core.trades[-1]["reason"], "segment_end")
        self.assertGreater(result.realized_r, 0)

    def test_target_exposure_resizes_and_observes_reward_state(self) -> None:
        decisions = pd.DataFrame(
            {
                "Open": [100, 100, 100],
                "High": [100, 100, 100],
                "Low": [100, 100, 100],
                "Close": [100, 100, 100],
                "atr": [10, 10, 10],
                "feature": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2025-01-01 01:00", periods=3, freq="h", tz="UTC"),
        )
        m1 = pd.DataFrame(
            {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0},
            index=pd.date_range("2025-01-01 01:01", periods=120, freq="min", tz="UTC"),
        )
        environment = BracketTradingEnvV2(
            decisions,
            m1,
            ["feature"],
            action_mode="sac",
            randomize=False,
            commission_rate=0,
        )
        observation, _ = environment.reset()
        self.assertEqual(observation.shape, (8,))
        _, first_reward, _, _, first_info = environment.step(np.asarray([0.5], dtype=np.float32))
        _, second_reward, _, truncated, second_info = environment.step(np.asarray([0.8], dtype=np.float32))
        self.assertTrue(np.isfinite(first_reward))
        self.assertTrue(np.isfinite(second_reward))
        self.assertEqual(first_info["target_exposure"], 0.5)
        self.assertTrue(truncated)
        self.assertEqual(second_info["target_exposure"], 0.0)
        self.assertEqual(environment.core.trades[0]["reason"], "signal")

    def test_native_puffer_environment_batches_and_resets_segments(self) -> None:
        decisions = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
                "atr": [1.0] * 5,
                "feature": [0.0] * 5,
            },
            index=pd.date_range("2025-01-01 01:00", periods=5, freq="h", tz="UTC"),
        )
        m1 = pd.DataFrame(
            {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0},
            index=pd.date_range("2025-01-01 01:01", periods=240, freq="min", tz="UTC"),
        )
        environment = PufferTradingEnv(
            (decisions,),
            m1,
            ["feature"],
            num_agents=2,
            episode_steps=2,
            random_seed=7,
            commission_rate=0.0,
            base_spread_bps=0.0,
            perturbation_config=PerturbationConfig(),
        )
        observations, _ = environment.reset(seed=7)
        self.assertEqual(observations.shape, (2, 8))
        self.assertIs(environment.envs[0].m1_bars, m1)
        self.assertNotEqual(
            environment.envs[0].decision_bars.index[0],
            environment.envs[1].decision_bars.index[0],
        )
        self.assertTrue(
            np.shares_memory(
                environment.envs[0].core._m1_open,
                environment.envs[1].core._m1_open,
            )
        )
        actions = np.zeros((2, 1), dtype=np.float32)
        environment.step(actions)
        _, _, terminals, truncations, infos = environment.step(actions)
        self.assertTrue(bool(terminals.all()))
        self.assertFalse(bool(truncations.any()))
        self.assertEqual(len(infos), 2)


if __name__ == "__main__":
    unittest.main()
