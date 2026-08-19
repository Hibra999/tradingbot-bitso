from __future__ import annotations

import unittest

import numpy as np

from rl import RecurrentPolicyRunner, lower_tail_scores


class _FakeRecurrentModel:
    def __init__(self):
        self.calls = []

    def predict(self, observation, *, state, episode_start, deterministic):
        self.calls.append((state, episode_start.copy(), deterministic))
        return np.array([1]), f"state-{len(self.calls)}"


class CandidateTests(unittest.TestCase):
    def test_cvar_uses_lower_quantiles_instead_of_ordinary_mean(self) -> None:
        quantiles = np.ones((1, 10, 2))
        quantiles[0, 0, 0] = 0
        quantiles[0, 1:, 0] = 100
        self.assertEqual(int(quantiles.mean(axis=1).argmax(axis=1)[0]), 0)
        self.assertEqual(int(lower_tail_scores(quantiles, 0.1).argmax(axis=1)[0]), 1)

    def test_recurrent_runner_carries_and_resets_lstm_state(self) -> None:
        model = _FakeRecurrentModel()
        runner = RecurrentPolicyRunner(model)
        runner.predict(np.zeros(3))
        runner.predict(np.zeros(3))
        runner.predict(np.zeros(3), episode_start=True)
        self.assertTrue(model.calls[0][1][0])
        self.assertEqual(model.calls[1][0], "state-1")
        self.assertIsNone(model.calls[2][0])
        self.assertTrue(model.calls[2][1][0])


if __name__ == "__main__":
    unittest.main()
