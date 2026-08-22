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

from .actions import TARGET_EXPOSURE_LEVELS


PUFFER_ALGORITHM = "pufferl"
PUFFER_AGENT_NAME = "PuffeRL-LSTM"
PUFFER_ACTION_ENCODING = "categorical_alpha_residual_expectation_11_v2"
PUFFER_MODAL_RESIDUAL_ACTION_ENCODING = "categorical_alpha_residual_11_v1"
PUFFER_ABSOLUTE_ACTION_ENCODING = "categorical_target_exposure_11_v1"
PUFFER_LEGACY_ACTION_ENCODING = "normal_target_exposure_v1"


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


def build_puffer_policy(
    env: Any,
    *,
    device: str | None = None,
    action_encoding: str = PUFFER_ACTION_ENCODING,
) -> nn.Module:
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
    if action_encoding in {PUFFER_ACTION_ENCODING, PUFFER_MODAL_RESIDUAL_ACTION_ENCODING}:
        decoder = policy.policy.decoder
        nn.init.zeros_(decoder.weight)
        nn.init.zeros_(decoder.bias)
        with torch.no_grad():
            decoder.bias[len(TARGET_EXPOSURE_LEVELS) // 2] = 3.0
    elif action_encoding not in {
        PUFFER_ABSOLUTE_ACTION_ENCODING,
        PUFFER_LEGACY_ACTION_ENCODING,
    }:
        raise ValueError(f"unsupported PuffeRL action encoding: {action_encoding}")
    policy.action_space = env.single_action_space
    policy.action_encoding = action_encoding
    return policy.to(device or _torch_device())


def load_puffer_policy(
    path: str | Path,
    observation_size: int,
    *,
    device: str | None = None,
    action_encoding: str = PUFFER_LEGACY_ACTION_ENCODING,
) -> nn.Module:
    selected_device = device or _torch_device()
    observation_space = spaces.Box(
        -np.inf,
        np.inf,
        shape=(observation_size,),
        dtype=np.float32,
    )
    if action_encoding in {
        PUFFER_ACTION_ENCODING,
        PUFFER_MODAL_RESIDUAL_ACTION_ENCODING,
        PUFFER_ABSOLUTE_ACTION_ENCODING,
    }:
        action_space = spaces.Discrete(len(TARGET_EXPOSURE_LEVELS))
    elif action_encoding == PUFFER_LEGACY_ACTION_ENCODING:
        action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    else:
        raise ValueError(f"unsupported PuffeRL action encoding: {action_encoding}")
    specification = SimpleNamespace(
        single_observation_space=observation_space,
        observation_space=observation_space,
        single_action_space=action_space,
    )
    policy = build_puffer_policy(
        specification,
        device=selected_device,
        action_encoding=action_encoding,
    )
    state = torch.load(Path(path), map_location=selected_device, weights_only=True)
    policy.load_state_dict(state)
    return policy.eval()


class PufferPolicyRunner:
    def __init__(self, policy: nn.Module):
        self.policy = policy.eval()
        self.policy_action_space = policy.action_space
        self.action_encoding = getattr(
            policy,
            "action_encoding",
            PUFFER_ABSOLUTE_ACTION_ENCODING,
        )
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.device = next(policy.parameters()).device
        self._positive_residual_levels = torch.arange(
            1,
            len(TARGET_EXPOSURE_LEVELS) // 2 + 1,
            device=self.device,
        ) / (len(TARGET_EXPOSURE_LEVELS) - 1)
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
            logits, _ = self.policy.forward_eval(values, self.state)
            if isinstance(logits, torch.Tensor):
                if logits.shape[-1] != len(TARGET_EXPOSURE_LEVELS):
                    raise ValueError("PuffeRL categorical action size is invalid")
                if deterministic and self.action_encoding == PUFFER_ACTION_ENCODING:
                    probabilities = torch.softmax(logits, dim=-1)
                    center = len(TARGET_EXPOSURE_LEVELS) // 2
                    residual = (
                        (
                            probabilities[..., center + 1 :]
                            - torch.flip(probabilities[..., :center], dims=(-1,))
                        )
                        * self._positive_residual_levels
                    ).sum(dim=-1)
                else:
                    action_index = (
                        logits.argmax(dim=-1)
                        if deterministic
                        else torch.distributions.Categorical(logits=logits).sample()
                    )
                    residual = (
                        action_index.to(torch.float32) / (len(TARGET_EXPOSURE_LEVELS) - 1)
                        - 0.5
                    )
                if self.action_encoding in {
                    PUFFER_ACTION_ENCODING,
                    PUFFER_MODAL_RESIDUAL_ACTION_ENCODING,
                }:
                    if observation.size < 8:
                        raise ValueError("residual PuffeRL observation is missing alpha exposure")
                    action = residual + float(observation[-8])
                else:
                    action = residual + 0.5
            else:
                action = logits.loc if deterministic else logits.sample()
        result = action.clamp(0.0, 1.0).cpu().numpy()
        return result.reshape(-1)
