from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


def aggregate_seed_results(results: list[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not results:
        raise ValueError("at least one seed result is required")
    metrics = sorted(set.intersection(*(set(result) for result in results)))
    output: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([result[metric] for result in results], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            q1, q3 = np.percentile(values, [25, 75])
            ordered = np.sort(values)
            trim = int(np.floor(len(ordered) * 0.25))
            middle = ordered[trim : len(ordered) - trim] if trim else ordered
            rng = np.random.default_rng(0)
            bootstrap = rng.choice(values, size=(2_000, len(values)), replace=True).mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
            output[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "iqr": float(q3 - q1),
                "iqm": float(middle.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
            }
    return output


@dataclass(frozen=True)
class SeedEvaluation:
    seeds: tuple[int, ...]
    results: tuple[dict[str, float], ...]
    aggregate: dict[str, dict[str, float]]
    sri_pass: bool

    def manifest(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "results": list(self.results),
            "aggregate": self.aggregate,
            "sri": {"definition": "mean_dsr_z > 0 and mean_sharpe > 1", "pass": self.sri_pass},
        }


class SeedHarness:
    def __init__(self, seeds: tuple[int, ...], *, smoke: bool = False):
        if not seeds:
            raise ValueError("seed evaluation requires at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        self.seeds = seeds

    def run(self, evaluator: Callable[[int], Mapping[str, float]]) -> SeedEvaluation:
        results = tuple(dict(evaluator(seed)) for seed in self.seeds)
        aggregate = aggregate_seed_results(list(results))
        sharpe = aggregate.get("sharpe", {}).get("mean", float("-inf"))
        dsr_z = aggregate.get("dsr_z", {}).get("mean", float("-inf"))
        return SeedEvaluation(self.seeds, results, aggregate, dsr_z > 0 and sharpe > 1)
