from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from rl import PufferPolicyRunner, build_puffer_policy


class _FakePufferPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def forward_eval(self, observations, state):
        previous = state["lstm_h"]
        mean = torch.full((1, 1), 0.25, device=observations.device)
        if previous is not None:
            mean += previous
        state["lstm_h"] = mean
        state["lstm_c"] = mean
        return torch.distributions.Normal(mean, torch.ones_like(mean)), torch.zeros(1, 1)


class _FakeCategoricalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.action_space = spaces.Discrete(11)

    def forward_eval(self, observations, state):
        logits = torch.zeros((1, 11), device=observations.device)
        logits[:, 7] = 1.0
        return logits, torch.zeros(1, 1)


class CandidateTests(unittest.TestCase):
    def test_puffer_policy_resets_lstm_state_on_episode_end(self) -> None:
        observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        environment = SimpleNamespace(
            single_observation_space=observation_space,
            observation_space=observation_space,
            single_action_space=spaces.Discrete(11),
        )
        policy = build_puffer_policy(environment, device="cpu")
        state = {"lstm_h": None, "lstm_c": None, "done": torch.zeros(1, dtype=torch.bool)}
        policy.forward_eval(torch.ones(1, 3), state)
        self.assertFalse(
            bool(torch.allclose(state["lstm_h"], torch.zeros_like(state["lstm_h"])))
        )
        state["done"] = torch.ones(1, dtype=torch.bool)
        policy.forward_eval(torch.zeros(1, 3), state)
        self.assertTrue(
            bool(torch.allclose(state["lstm_h"], torch.zeros_like(state["lstm_h"])))
        )

    def test_puffer_runner_carries_and_resets_lstm_state(self) -> None:
        runner = PufferPolicyRunner(_FakePufferPolicy())
        self.assertAlmostEqual(float(runner.predict(np.zeros(3))[0]), 0.25)
        self.assertAlmostEqual(float(runner.predict(np.zeros(3))[0]), 0.50)
        self.assertAlmostEqual(
            float(runner.predict(np.zeros(3), episode_start=True)[0]),
            0.25,
        )

    def test_puffer_runner_maps_categorical_policy_to_public_exposure(self) -> None:
        runner = PufferPolicyRunner(_FakeCategoricalPolicy())
        self.assertEqual(runner.action_space.shape, (1,))
        self.assertAlmostEqual(float(runner.predict(np.zeros(3))[0]), 0.7)


if __name__ == "__main__":
    unittest.main()
