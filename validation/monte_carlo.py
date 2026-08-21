from __future__ import annotations
import tempfile
from dataclasses import dataclass
import numpy as np, pandas as pd
from tqdm import tqdm

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
    if len(values) < block_size or bool((values <= -1).any()): raise ValueError("returns must be finite, greater than -1, and cover one block")
    if paths < 1 or horizon < 1 or batch_size < 1: raise ValueError("paths, horizon, and batch_size must be positive")

    rng = np.random.default_rng(seed)
    max_dd = np.empty(paths)
    rec_dur = np.empty(paths)
    terminal = np.empty(paths)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)

    with tempfile.TemporaryDirectory(prefix="tradingbot-mc-") as folder:
        path_mat = np.memmap(f"{folder}/equity.dat", mode="w+", dtype="float32", shape=(paths, horizon))
        block_cnt = int(np.ceil(horizon / block_size))
        offsets = np.arange(block_size)
        total_batches = (paths + batch_size - 1) // batch_size
        for start in tqdm(range(0, paths, batch_size), total=total_batches, desc="Monte Carlo paths", unit="batch", leave=False):
            stop = min(start + batch_size, paths)
            count = stop - start
            block_starts = rng.integers(0, len(values) - block_size + 1, size=(count, block_cnt))
            sampled = values[(block_starts[..., None] + offsets).reshape(count, -1)[:, :horizon]]
            equity = initial_equity * np.cumprod(1 + sampled, axis=1)
            peaks = np.maximum.accumulate(equity, axis=1)
            drawdown = equity / peaks - 1
            max_dd[start:stop] = drawdown.min(axis=1)
            terminal[start:stop] = equity[:, -1]

            is_dd = (drawdown < 0).T
            curr = np.zeros(count, dtype=int)
            long = np.zeros(count, dtype=int)
            for col in is_dd:
                curr = np.where(col, curr + 1, 0)
                np.maximum(long, curr, out=long)
            rec_dur[start:stop] = long
            path_mat[start:stop] = equity.astype("float32")
        path_mat.flush()

        cone = np.empty((len(quantiles), horizon), dtype=float)
        total_col_batches = (horizon + 255) // 256
        for col_s in tqdm(range(0, horizon, 256), total=total_col_batches, desc="Equity cone", unit="col_batch", leave=False):
            col_e = min(col_s + 256, horizon)
            cone[:, col_s:col_e] = np.quantile(np.asarray(path_mat[:, col_s:col_e]), quantiles, axis=0)
        equity_cone = pd.DataFrame(cone.T, index=pd.RangeIndex(1, horizon + 1, name="step"), columns=[f"p{int(q * 100):02d}" for q in quantiles])
        path_mat._mmap.close()

    return MonteCarloResult(
        equity_cone=equity_cone,
        terminal_equity=_summary(terminal),
        maximum_drawdown=_summary(max_dd),
        recovery_duration=_summary(rec_dur),
        ruin_probability_20=float(np.mean(max_dd <= -0.20)),
        ruin_probability_30=float(np.mean(max_dd <= -0.30)),
        seed=seed,
        paths=paths,
        block_size=block_size,
    )
