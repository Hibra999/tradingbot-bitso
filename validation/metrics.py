from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass(frozen=True)
class ProbabilityRatio:
    z_score: float
    probability: float
    p_value: float

def _returns(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: raise ValueError("at least two finite returns are required")
    return np.clip(arr, -1.0, None)

def drawdowns(returns: np.ndarray | list[float]) -> np.ndarray:
    v = _returns(returns)
    eq_curve = np.cumprod(np.maximum(1.0 + v, 0.0))
    peak = np.maximum.accumulate(eq_curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, eq_curve / peak - 1.0, -1.0)
    return dd

def sharpe_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    try: v = _returns(returns)
    except ValueError: return float("nan")
    vol = v.std(ddof=1)
    return float(v.mean() / vol * np.sqrt(periods_per_year)) if vol > 1e-12 else float("nan")

def sortino_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    try: v = _returns(returns)
    except ValueError: return float("nan")
    ds = np.sqrt(np.mean(np.minimum(v, 0) ** 2))
    return float(v.mean() / ds * np.sqrt(periods_per_year)) if ds > 1e-12 else float("nan")

def calmar_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    try: v = _returns(returns)
    except ValueError: return float("nan")
    cum_ret = float(np.prod(np.maximum(1.0 + v, 0.0)))
    ann = float(cum_ret ** (periods_per_year / len(v)) - 1) if cum_ret > 0 else -1.0
    mdd = abs(float(drawdowns(v).min()))
    return ann / mdd if mdd > 1e-12 else float("nan")

def probabilistic_sharpe_ratio(returns: np.ndarray | list[float], benchmark_sharpe: float = 0.0, periods_per_year: int = 365 * 24) -> ProbabilityRatio:
    try: v = _returns(returns)
    except ValueError: return ProbabilityRatio(0.0, 0.5, 0.5)
    vol = v.std(ddof=1)
    if vol <= 1e-12: return ProbabilityRatio(0.0, 0.5, 0.5)
    sample = v.mean() / vol
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    skew = stats.skew(v, bias=False)
    kurt = stats.kurtosis(v, fisher=False, bias=False)
    denom = np.sqrt(max(1 - skew * sample + ((kurt - 1) / 4) * sample ** 2, 1e-15))
    z = float((sample - benchmark) * np.sqrt(len(v) - 1) / denom)
    prob = float(stats.norm.cdf(z))
    return ProbabilityRatio(z, prob, 1 - prob)

def deflated_sharpe_ratio(returns: np.ndarray | list[float], trial_sharpes: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> ProbabilityRatio:
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    if len(trials) < 2: return probabilistic_sharpe_ratio(returns, 0.0, periods_per_year)
    trial_std = trials.std(ddof=1)
    if trial_std <= 1e-12: emax = float(trials.mean())
    else:
        gamma = 0.5772156649015329
        emax = trials.mean() + trial_std * ((1 - gamma) * stats.norm.ppf(1 - 1 / len(trials)) + gamma * stats.norm.ppf(1 - 1 / (len(trials) * np.e)))
    return probabilistic_sharpe_ratio(returns, float(emax), periods_per_year)

def centered_block_bootstrap_test(returns: np.ndarray | list[float], *, block_size: int = 24, repetitions: int = 2_000, seed: int = 0) -> dict[str, float]:
    v = _returns(returns)
    n = len(v)
    if not 1 <= block_size <= n or repetitions < 1: raise ValueError("invalid block size or repetition count")
    cnt = v - v.mean()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=(repetitions, nb))
    idx = (starts[..., None] + np.arange(block_size)).reshape(repetitions, -1)[:, :n]
    means = cnt[idx].mean(axis=1)
    obs = float(v.mean())
    return {"mean": obs, "p_value": float((1 + np.count_nonzero(np.abs(means) >= abs(obs))) / (repetitions + 1))}

def institutional_metrics(returns: np.ndarray | list[float], *, trial_sharpes: np.ndarray | list[float], periods_per_year: int = 365 * 24, bootstrap_repetitions: int = 2_000, seed: int = 0) -> dict[str, float]:
    v = _returns(returns)
    gains, losses = float(v[v > 0].sum()), float(-v[v < 0].sum())
    q05, q95 = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
    psr = probabilistic_sharpe_ratio(v, periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(v, trial_sharpes, periods_per_year)
    t_test = stats.ttest_1samp(v, popmean=0.0)
    boot = centered_block_bootstrap_test(v, block_size=min(24, len(v)), repetitions=bootstrap_repetitions, seed=seed)
    res = {
        "sharpe": sharpe_ratio(v, periods_per_year),
        "sortino": sortino_ratio(v, periods_per_year),
        "calmar": calmar_ratio(v, periods_per_year),
        "omega": float(gains / losses) if losses > 0 else float("inf"),
        "gain_to_pain": float(v.sum() / losses) if losses > 0 else float("inf"),
        "tail_ratio": float(q95 / abs(q05)) if q05 < 0 else float("inf"),
        "psr_probability": psr.probability,
        "psr_p_value": psr.p_value,
        "dsr_z": dsr.z_score,
        "dsr_probability": dsr.probability,
        "dsr_p_value": dsr.p_value,
        "t_statistic": float(t_test.statistic) if np.isfinite(t_test.statistic) else 0.0,
        "t_test_p_value": float(t_test.pvalue) if np.isfinite(t_test.pvalue) else 1.0,
        "bootstrap_p_value": boot["p_value"],
        "max_drawdown": float(drawdowns(v).min()),
    }
    for conf in (0.95, 0.99):
        q = float(np.quantile(v, 1 - conf))
        suf = int(conf * 100)
        res[f"var_{suf}"] = float(-q)
        tail_mean = float(-v[v <= q].mean()) if (v <= q).any() else float(-q)
        res[f"cvar_{suf}"] = tail_mean
    return res
