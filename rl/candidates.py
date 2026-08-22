from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn


PUFFER_ALGORITHM = "pufferl"
PUFFER_AGENT_NAME = "PuffeRL-LSTM"


@lru_cache(maxsize=1)
def _torch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return "cuda"


@lru_cache(maxsize=1)
def require_pufferlib() -> None:
    try:
        installed = version("pufferlib")
    except PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned PufferLib dependencies from requirements.txt") from exc
    if installed != "3.0.0":
        raise RuntimeError(f"PuffeRL is pinned to PufferLib 3.0.0, found {installed}")


def build_puffer_policy(env: Any, *, device: str | None = None) -> nn.Module:
    require_pufferlib()
    from pufferlib.models import Default, LSTMWrapper

    class EpisodeAwareLSTMWrapper(LSTMWrapper):
        def forward_eval(self, observations, state):
            done = state.get("done")
            if done is not None and state.get("lstm_h") is not None:
                keep = (~done.bool()).unsqueeze(-1)
                state["lstm_h"] = state["lstm_h"] * keep
                state["lstm_c"] = state["lstm_c"] * keep
            return super().forward_eval(observations, state)

    policy = EpisodeAwareLSTMWrapper(
        env,
        Default(env, hidden_size=128),
        input_size=128,
        hidden_size=128,
    )
    policy.action_space = env.single_action_space
    return policy.to(device or _torch_device())


def load_puffer_policy(
    path: str | Path,
    observation_size: int,
    *,
    device: str | None = None,
) -> nn.Module:
    selected_device = device or _torch_device()
    observation_space = spaces.Box(
        -np.inf,
        np.inf,
        shape=(observation_size,),
        dtype=np.float32,
    )
    specification = SimpleNamespace(
        single_observation_space=observation_space,
        observation_space=observation_space,
        single_action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    )
    policy = build_puffer_policy(specification, device=selected_device)
    state = torch.load(Path(path), map_location=selected_device, weights_only=True)
    policy.load_state_dict(state)
    return policy.eval()


class PufferPolicyRunner:
    def __init__(self, policy: nn.Module):
        self.policy = policy.eval()
        self.action_space = policy.action_space
        self.device = next(policy.parameters()).device
        self.reset()

    def reset(self) -> None:
        self.state: dict[str, torch.Tensor | None] = {"lstm_h": None, "lstm_c": None}

    def predict(
        self,
        observation: np.ndarray,
        *,
        episode_start: bool = False,
        deterministic: bool = True,
    ) -> np.ndarray:
        if episode_start:
            self.reset()
        values = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.inference_mode():
            distribution, _ = self.policy.forward_eval(values, self.state)
            action = distribution.loc if deterministic else distribution.sample()
        result = action.clamp(0.0, 1.0).cpu().numpy()
        return result[0]
