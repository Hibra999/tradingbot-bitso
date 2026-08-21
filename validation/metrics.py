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
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2 or bool((array <= -1).any()):
        raise ValueError("at least two finite returns greater than -1 are required")
    return array


def drawdowns(returns: np.ndarray | list[float]) -> np.ndarray:
    equity = np.cumprod(1 + _returns(returns))
    return equity / np.maximum.accumulate(equity) - 1


def sharpe_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    values = _returns(returns)
    volatility = values.std(ddof=1)
    return float(values.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 0 else float("nan")


def sortino_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    values = _returns(returns)
    downside = np.sqrt(np.mean(np.minimum(values, 0) ** 2))
    return float(values.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else float("nan")


def calmar_ratio(returns: np.ndarray | list[float], periods_per_year: int = 365 * 24) -> float:
    values = _returns(returns)
    annualized = float(np.prod(1 + values) ** (periods_per_year / len(values)) - 1)
    maximum_drawdown = abs(float(drawdowns(values).min()))
    return annualized / maximum_drawdown if maximum_drawdown > 0 else float("nan")


def advanced_metrics(
    returns: np.ndarray | list[float], periods_per_year: int = 365 * 24
) -> dict[str, float]:
    values = _returns(returns)
    volatility = values.std(ddof=1)
    gains, losses = values[values > 0], values[values < 0]
    average_loss = abs(float(losses.mean())) if len(losses) else 0.0
    dd = drawdowns(values)
    underwater = dd < 0
    changes = np.diff(np.pad(underwater.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    durations = ends - starts
    episode_depths = -np.minimum.reduceat(dd, starts) if len(starts) else np.array([], dtype=float)
    expectancy = float(values.mean())
    return {
        "sharpe": sharpe_ratio(values, periods_per_year),
        "sortino": sortino_ratio(values, periods_per_year),
        "calmar": calmar_ratio(values, periods_per_year),
        "sqn": float(np.sqrt(len(values)) * expectancy / volatility) if volatility > 0 else float("nan"),
        "expectancy": expectancy,
        "expectancy_ratio": expectancy / average_loss if average_loss > 0 else float("inf"),
        "profit_factor": float(gains.sum() / -losses.sum()) if len(losses) else float("inf"),
        "win_rate": float(np.mean(values > 0)),
        "max_drawdown": float(dd.min()),
        "average_drawdown": float(episode_depths.mean()) if len(episode_depths) else 0.0,
        "drawdown_duration_max": float(durations.max()) if len(durations) else 0.0,
        "drawdown_duration_mean": float(durations.mean()) if len(durations) else 0.0,
    }


def probabilistic_sharpe_ratio(
    returns: np.ndarray | list[float],
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 365 * 24,
) -> ProbabilityRatio:
    values = _returns(returns)
    sample = values.mean() / values.std(ddof=1)
    benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    skew = stats.skew(values, bias=False)
    kurtosis = stats.kurtosis(values, fisher=False, bias=False)
    denominator = np.sqrt(max(1 - skew * sample + ((kurtosis - 1) / 4) * sample**2, 1e-15))
    z_score = float((sample - benchmark) * np.sqrt(len(values) - 1) / denominator)
    probability = float(stats.norm.cdf(z_score))
    return ProbabilityRatio(z_score, probability, 1 - probability)


def deflated_sharpe_ratio(
    returns: np.ndarray | list[float],
    trial_sharpes: np.ndarray | list[float],
    periods_per_year: int = 365 * 24,
) -> ProbabilityRatio:
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    if len(trials) < 2:
        raise ValueError("DSR requires at least two trial Sharpe ratios")
    gamma = 0.5772156649015329
    expected_max = trials.mean() + trials.std(ddof=1) * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / len(trials))
        + gamma * stats.norm.ppf(1 - 1 / (len(trials) * np.e))
    )
    return probabilistic_sharpe_ratio(returns, float(expected_max), periods_per_year)


def centered_block_bootstrap_test(
    returns: np.ndarray | list[float],
    *,
    block_size: int = 24,
    repetitions: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    values = _returns(returns)
    if not 1 <= block_size <= len(values) or repetitions < 1:
        raise ValueError("invalid block size or repetition count")
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(values) / block_size))
    starts = rng.integers(0, len(values) - block_size + 1, size=(repetitions, block_count))
    offsets = np.arange(block_size)
    samples = centered[(starts[..., None] + offsets).reshape(repetitions, -1)[:, : len(values)]]
    bootstrap_means = samples.mean(axis=1)
    observed = float(values.mean())
    p_value = float((1 + np.count_nonzero(np.abs(bootstrap_means) >= abs(observed))) / (repetitions + 1))
    return {"mean": observed, "p_value": p_value}


def probability_of_backtest_overfitting(
    selection_scores: np.ndarray | list[list[float]],
    evaluation_scores: np.ndarray | list[list[float]],
) -> dict[str, float]:
    selection = np.asarray(selection_scores, dtype=float)
    evaluation = np.asarray(evaluation_scores, dtype=float)
    if selection.shape != evaluation.shape or selection.ndim != 2:
        raise ValueError("PBO score matrices must be aligned and two-dimensional")
    if selection.shape[0] < 2 or selection.shape[1] < 2:
        raise ValueError("PBO requires at least two folds and two configurations")
    logits: list[float] = []
    for in_sample, out_of_sample in zip(selection, evaluation):
        finite = np.isfinite(in_sample) & np.isfinite(out_of_sample)
        if np.count_nonzero(finite) < 2:
            continue
        selected = int(np.nanargmax(np.where(finite, in_sample, np.nan)))
        ranks = stats.rankdata(out_of_sample[finite], method="average")
        selected_rank = float(ranks[np.flatnonzero(np.flatnonzero(finite) == selected)[0]])
        relative_rank = selected_rank / (len(ranks) + 1)
        logits.append(float(np.log(relative_rank / (1 - relative_rank))))
    if not logits:
        raise ValueError("PBO has no folds with finite aligned scores")
    values = np.asarray(logits)
    return {
        "pbo_probability": float(np.mean(values <= 0)),
        "pbo_median_logit": float(np.median(values)),
        "pbo_folds": float(len(values)),
    }


def institutional_metrics(
    returns: np.ndarray | list[float],
    *,
    trial_sharpes: np.ndarray | list[float],
    periods_per_year: int = 365 * 24,
    bootstrap_repetitions: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    values = _returns(returns)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    q05, q95 = np.quantile(values, [0.05, 0.95])
    psr = probabilistic_sharpe_ratio(values, periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(values, trial_sharpes, periods_per_year)
    t_test = stats.ttest_1samp(values, popmean=0.0)
    bootstrap = centered_block_bootstrap_test(
        values,
        block_size=min(24, len(values)),
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    result = {
        "sharpe": sharpe_ratio(values, periods_per_year),
        "sortino": sortino_ratio(values, periods_per_year),
        "calmar": calmar_ratio(values, periods_per_year),
        "omega": float(gains / losses) if losses > 0 else float("inf"),
        "gain_to_pain": float(values.sum() / losses) if losses > 0 else float("inf"),
        "tail_ratio": float(q95 / abs(q05)) if q05 < 0 else float("inf"),
        "psr_probability": psr.probability,
        "psr_p_value": psr.p_value,
        "dsr_z": dsr.z_score,
        "dsr_probability": dsr.probability,
        "dsr_p_value": dsr.p_value,
        "t_statistic": float(t_test.statistic),
        "t_test_p_value": float(t_test.pvalue),
        "bootstrap_p_value": bootstrap["p_value"],
        "max_drawdown": float(drawdowns(values).min()),
    }
    result.update(advanced_metrics(values, periods_per_year))
    for confidence in (0.95, 0.99):
        tail = values[values <= np.quantile(values, 1 - confidence)]
        suffix = int(confidence * 100)
        result[f"var_{suffix}"] = float(-np.quantile(values, 1 - confidence))
        result[f"cvar_{suffix}"] = float(-tail.mean())
    return result
