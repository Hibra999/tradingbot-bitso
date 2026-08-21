from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def _torch_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return "cuda"


def _require_pinned_stack() -> None:
    try:
        installed = (version("stable-baselines3"), version("sb3-contrib"))
    except PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned RL dependencies from requirements.txt") from exc
    if installed != ("2.9.0", "2.9.0"):
        raise RuntimeError(f"CVaRQRDQN is pinned to SB3/SB3-Contrib 2.9.0, found {installed}")


def lower_tail_scores(quantiles: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    values = np.asarray(quantiles, dtype=float)
    if values.ndim != 3 or not 0 < alpha <= 1:
        raise ValueError("quantiles must have shape (batch, quantiles, actions) and alpha in (0, 1]")
    tail_count = max(1, int(np.ceil(values.shape[1] * alpha)))
    return values[:, :tail_count, :].mean(axis=1)


def build_recurrent_ppo(env: Any, *, seed: int, **overrides: Any):
    _require_pinned_stack()
    from sb3_contrib import RecurrentPPO

    n_steps = 256
    rollout_size = n_steps * getattr(env, "num_envs", 1)
    batch_size = min(1_024, rollout_size)
    while rollout_size % batch_size:
        batch_size //= 2
    options = {
        "learning_rate": 3e-4,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "policy_kwargs": {
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "net_arch": [128, 64],
            "optimizer_kwargs": {"foreach": True},
        },
        "device": _torch_device(),
        "seed": seed,
        "verbose": 0,
    }
    options.update(overrides)
    return RecurrentPPO("MlpLstmPolicy", env, **options)


def build_sac(env: Any, *, seed: int, **overrides: Any):
    _require_pinned_stack()
    from stable_baselines3 import SAC

    batch_size = min(512, max(256, 64 * getattr(env, "num_envs", 1)))
    options = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": batch_size,
        "train_freq": (16, "step"),
        "gradient_steps": 16,
        "policy_kwargs": {"optimizer_kwargs": {"foreach": True}},
        "device": _torch_device(),
        "seed": seed,
        "verbose": 0,
    }
    options.update(overrides)
    return SAC("MlpPolicy", env, **options)


def build_tqc(env: Any, *, seed: int, **overrides: Any):
    _require_pinned_stack()
    from sb3_contrib import TQC

    batch_size = min(512, max(256, 64 * getattr(env, "num_envs", 1)))
    options = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": batch_size,
        "train_freq": (16, "step"),
        "gradient_steps": 16,
        "top_quantiles_to_drop_per_net": 2,
        "policy_kwargs": {
            "n_critics": 5,
            "n_quantiles": 25,
            "optimizer_kwargs": {"foreach": True},
        },
        "device": _torch_device(),
        "seed": seed,
        "verbose": 0,
    }
    options.update(overrides)
    return TQC("MlpPolicy", env, **options)


class RecurrentPolicyRunner:
    def __init__(self, model: Any):
        self.model = model
        self.state = None
        self.episode_start = True

    def reset(self) -> None:
        self.state = None
        self.episode_start = True

    def predict(self, observation: np.ndarray, *, episode_start: bool = False, deterministic: bool = True):
        if episode_start:
            self.reset()
        action, self.state = self.model.predict(
            observation,
            state=self.state,
            episode_start=np.asarray([self.episode_start], dtype=bool),
            deterministic=deterministic,
        )
        self.episode_start = False
        return action


try:
    import torch as th
    from sb3_contrib import QRDQN
    from sb3_contrib.common.utils import quantile_huber_loss
except ModuleNotFoundError:  # Lightweight VPS checks do not install the GPU/RL stack.
    th = None
    QRDQN = object  # type: ignore[assignment,misc]


class CVaRQRDQN(QRDQN):  # type: ignore[misc,valid-type]
    """Pinned QR-DQN using the lower-alpha quantile mean for behavior and targets."""

    def __init__(self, *args: Any, cvar_alpha: float = 0.10, **kwargs: Any):
        _require_pinned_stack()
        if th is None:
            raise RuntimeError("Install the pinned RL dependencies from requirements.txt")
        if not 0 < cvar_alpha <= 1:
            raise ValueError("cvar_alpha must be in (0, 1]")
        self.cvar_alpha = cvar_alpha
        super().__init__(*args, **kwargs)

    def _tail_count(self) -> int:
        return max(1, int(np.ceil(self.n_quantiles * self.cvar_alpha)))

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            vectorized = self.policy.is_vectorized_observation(observation)
            count = observation[next(iter(observation))].shape[0] if vectorized and isinstance(observation, dict) else (observation.shape[0] if vectorized else 1)
            action = np.asarray([self.action_space.sample() for _ in range(count)])
            return (action if vectorized else action[0]), state
        observation_tensor, vectorized = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            quantiles = self.quantile_net(observation_tensor)
            action = quantiles[:, : self._tail_count(), :].mean(dim=1).argmax(dim=1).cpu().numpy()
        return (action if vectorized else action[0]), state

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        tail_count, losses = self._tail_count(), []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                next_actions = next_quantiles[:, :tail_count, :].mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                next_actions = next_actions.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(dim=2, index=next_actions).squeeze(dim=2)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles
            current_quantiles = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)
            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            losses.append(loss.detach())
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", th.stack(losses).mean().item() if losses else float("nan"))


def build_cvar_qrdqn(env: Any, *, seed: int, **overrides: Any) -> CVaRQRDQN:
    batch_size = min(256, max(64, 16 * getattr(env, "num_envs", 1)))
    options = {
        "learning_rate": 5e-5,
        "buffer_size": 100_000,
        "batch_size": batch_size,
        "train_freq": (32, "step"),
        "gradient_steps": 8,
        "policy_kwargs": {"n_quantiles": 200, "net_arch": [128, 64], "optimizer_kwargs": {"foreach": True}},
        "device": _torch_device(),
        "seed": seed,
        "verbose": 0,
    }
    options.update(overrides)
    return CVaRQRDQN("MlpPolicy", env, cvar_alpha=0.10, **options)
