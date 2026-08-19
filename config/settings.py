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


def required_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is missing")
    return value


@dataclass(frozen=True)
class BitsoConfig:
    rest_url: str = "https://api.bitso.com/v3"
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


@dataclass(frozen=True)
class ValidationConfig:
    temporal_groups: int = 6
    test_groups: int = 2
    embargo_bars: int = 200
    max_holding_bars: int = 24
    full_seeds: tuple[int, ...] = tuple(range(10))
    smoke_seeds: tuple[int, ...] = (0,)
    monte_carlo_paths: int = 5_000


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
        )
