from __future__ import annotations

import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    equity_cone: pd.DataFrame
    terminal_equity: dict[str, float]
    maximum_drawdown: dict[str, float]
    recovery_duration: dict[str, float]
    ruin_probability_20: float
    ruin_probability_30: float
    seed: int
    paths: int
    block_size: int

    def manifest(self) -> dict[str, object]:
        return {
            "terminal_equity": self.terminal_equity,
            "maximum_drawdown": self.maximum_drawdown,
            "recovery_duration": self.recovery_duration,
            "ruin_probability_20": self.ruin_probability_20,
            "ruin_probability_30": self.ruin_probability_30,
            "seed": self.seed,
            "paths": self.paths,
            "block_size": self.block_size,
        }


def _summary(values: np.ndarray) -> dict[str, float]:
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"mean": float(values.mean()), "p05": float(q05), "median": float(q50), "p95": float(q95)}


def moving_block_monte_carlo(
    returns: np.ndarray | list[float],
    *,
    paths: int = 5_000,
    horizon: int | None = None,
    block_size: int = 24,
    initial_equity: float = 1.0,
    batch_size: int = 128,
    seed: int = 0,
) -> MonteCarloResult:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    horizon = horizon or len(values)
    if len(values) < block_size or bool((values <= -1).any()):
        raise ValueError("returns must be finite, greater than -1, and cover one block")
    if paths < 1 or horizon < 1 or batch_size < 1:
        raise ValueError("paths, horizon, and batch_size must be positive")

    rng = np.random.default_rng(seed)
    maximum_drawdown = np.empty(paths)
    recovery_duration = np.empty(paths)
    terminal = np.empty(paths)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)

    with tempfile.TemporaryDirectory(prefix="tradingbot-mc-") as folder:
        path_matrix = np.memmap(f"{folder}/equity.dat", mode="w+", dtype="float32", shape=(paths, horizon))
        block_count = int(np.ceil(horizon / block_size))
        offsets = np.arange(block_size)
        for start in range(0, paths, batch_size):
            stop = min(start + batch_size, paths)
            count = stop - start
            block_starts = rng.integers(0, len(values) - block_size + 1, size=(count, block_count))
            sampled = values[(block_starts[..., None] + offsets).reshape(count, -1)[:, :horizon]]
            equity = initial_equity * np.cumprod(1 + sampled, axis=1)
            peaks = np.maximum.accumulate(equity, axis=1)
            drawdown = equity / peaks - 1
            maximum_drawdown[start:stop] = drawdown.min(axis=1)
            terminal[start:stop] = equity[:, -1]

            current = np.zeros(count, dtype=int)
            longest = np.zeros(count, dtype=int)
            for column in range(horizon):
                current = np.where(drawdown[:, column] < 0, current + 1, 0)
                longest = np.maximum(longest, current)
            recovery_duration[start:stop] = longest
            path_matrix[start:stop] = equity.astype("float32")
        path_matrix.flush()

        cone = np.empty((len(quantiles), horizon), dtype=float)
        for column_start in range(0, horizon, 256):
            column_stop = min(column_start + 256, horizon)
            cone[:, column_start:column_stop] = np.quantile(
                np.asarray(path_matrix[:, column_start:column_stop]), quantiles, axis=0
            )
        equity_cone = pd.DataFrame(
            cone.T,
            index=pd.RangeIndex(1, horizon + 1, name="step"),
            columns=[f"p{int(q * 100):02d}" for q in quantiles],
        )
        path_matrix._mmap.close()

    return MonteCarloResult(
        equity_cone=equity_cone,
        terminal_equity=_summary(terminal),
        maximum_drawdown=_summary(maximum_drawdown),
        recovery_duration=_summary(recovery_duration),
        ruin_probability_20=float(np.mean(maximum_drawdown <= -0.20)),
        ruin_probability_30=float(np.mean(maximum_drawdown <= -0.30)),
        seed=seed,
        paths=paths,
        block_size=block_size,
    )
