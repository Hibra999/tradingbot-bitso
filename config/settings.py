from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    try:
        result = default if value is None or not value.strip() else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def env_seeds(name: str, default: int) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return tuple(range(default))
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",")) if "," in value else tuple(range(int(value)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive count or comma-separated integers") from exc
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must define unique non-negative seeds")
    return seeds


def required_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is missing")
    return value


@dataclass(frozen=True)
class BitsoConfig:
    rest_url: str = "https://api.bitso.com/api/v3"
    websocket_url: str = "wss://ws.bitso.com"
    books: tuple[str, ...] = ("btc_usd", "eth_usd")
    public_requests_per_minute: int = 60
    private_requests_per_minute: int = 300
    request_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class RLConfig:
    algorithms: tuple[str, ...] = ("recurrent_ppo", "sac", "cvar_qrdqn")
    risk_fractions: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03)
    sl_atr_multipliers: tuple[float, ...] = (1.0, 1.5, 2.5, 3.5)
    tp_sl_ratios: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0)
    max_holding_bars: int = 24
    timesteps: dict[str, int] = field(
        default_factory=lambda: {"recurrent_ppo": 100_000, "sac": 100_000, "cvar_qrdqn": 100_000}
    )
    evaluations: int = 5
    recurrent_ppo_envs: int = 16

    @classmethod
    def from_env(cls) -> "RLConfig":
        names = {
            "recurrent_ppo": "RL_RECURRENT_PPO",
            "sac": "RL_SAC",
            "cvar_qrdqn": "RL_CVAR_QRDQN",
        }
        return cls(
            algorithms=tuple(name for name, prefix in names.items() if env_flag(f"{prefix}_ENABLED", True)),
            timesteps={name: env_int(f"{prefix}_TIMESTEPS", 100_000) for name, prefix in names.items()},
            evaluations=env_int("RL_EVALUATIONS", 5),
            recurrent_ppo_envs=env_int("RL_RECURRENT_PPO_ENVS", 16),
        )


@dataclass(frozen=True)
class ValidationConfig:
    temporal_groups: int = 3
    test_groups: int = 2
    embargo_bars: int = 200
    max_holding_bars: int = 24
    full_seeds: tuple[int, ...] = (0, 1)
    smoke_seeds: tuple[int, ...] = (0,)
    monte_carlo_paths: int = 5_000
    train_months: int = 36
    validation_months: int = 6
    evaluation_months: int = 6
    step_months: int = 6
    holdout_months: int = 6

    def __post_init__(self) -> None:
        if not 1 <= self.test_groups < self.temporal_groups:
            raise ValueError("test_groups must be in [1, temporal_groups)")
        if self.embargo_bars < 0 or self.max_holding_bars < 1 or not self.full_seeds or not self.smoke_seeds:
            raise ValueError("validation bars and seeds must be positive")
        if min(
            self.train_months,
            self.validation_months,
            self.evaluation_months,
            self.step_months,
            self.holdout_months,
        ) < 1:
            raise ValueError("walk-forward window lengths must be positive")

    @classmethod
    def from_env(cls) -> "ValidationConfig":
        return cls(
            temporal_groups=env_int("VALIDATION_TEMPORAL_GROUPS", 3, minimum=2),
            test_groups=env_int("VALIDATION_TEST_GROUPS", 2),
            full_seeds=env_seeds("VALIDATION_FULL_SEEDS", 2),
            embargo_bars=env_int("VALIDATION_EMBARGO_BARS", 200, minimum=0),
            monte_carlo_paths=env_int("VALIDATION_MONTE_CARLO_PATHS", 5_000),
            train_months=env_int("VALIDATION_TRAIN_MONTHS", 36),
            validation_months=env_int("VALIDATION_VAL_MONTHS", 6),
            evaluation_months=env_int("VALIDATION_EVAL_MONTHS", 6),
            step_months=env_int("VALIDATION_STEP_MONTHS", 6),
            holdout_months=env_int("VALIDATION_HOLDOUT_MONTHS", 6),
        )


@dataclass(frozen=True)
class RuntimeRiskParams:
    risk_fraction: Decimal = Decimal("0.005")
    max_drawdown_fraction: Decimal = Decimal("0.20")
    max_position_usd: Decimal = Decimal("1000")
    max_daily_loss_usd: Decimal = Decimal("100")

    def __post_init__(self) -> None:
        if not Decimal("0") < self.risk_fraction <= Decimal("0.03"):
            raise ValueError("risk_fraction must be in (0, 0.03]")
        if not Decimal("0") < self.max_drawdown_fraction < Decimal("1"):
            raise ValueError("max_drawdown_fraction must be in (0, 1)")
        if self.max_position_usd <= 0 or self.max_daily_loss_usd <= 0:
            raise ValueError("position and daily-loss limits must be positive")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (Path, Decimal)):
        return str(value)
    return value


@dataclass(frozen=True)
class AppConfig:
    profile: str = "smoke"
    research_symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    data_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    journal_path: Path = Path("data/execution.sqlite3")
    paper_mode: bool = True
    allow_margin_shorts: bool = False
    cache_only: bool = True
    bitso: BitsoConfig = field(default_factory=BitsoConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    risk: RuntimeRiskParams = field(default_factory=RuntimeRiskParams)

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be 'smoke' or 'full'")
        if not self.paper_mode and not env_flag("BITSO_LIVE_ENABLED"):
            raise ValueError("live mode requires BITSO_LIVE_ENABLED=true")
        if self.allow_margin_shorts and not env_flag("BITSO_MARGIN_SHORTS_ENABLED"):
            raise ValueError("margin shorts require BITSO_MARGIN_SHORTS_ENABLED=true")

    @property
    def promotable(self) -> bool:
        return self.profile == "full"

    def public_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_env(cls, *, profile: str = "smoke") -> "AppConfig":
        live = os.getenv("TRADING_MODE", "paper").strip().lower() == "live"
        return cls(
            profile=profile,
            paper_mode=not live,
            allow_margin_shorts=env_flag("BITSO_MARGIN_SHORTS_ENABLED"),
            cache_only=env_flag("CACHE_ONLY", True),
            rl=RLConfig.from_env(),
            validation=ValidationConfig.from_env(),
        )
