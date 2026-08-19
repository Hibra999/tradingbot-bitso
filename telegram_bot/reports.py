from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from validation import MonteCarloResult, drawdowns

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def generate_report(
    returns: pd.Series,
    monte_carlo: MonteCarloResult,
    destination: str | Path,
) -> Path:
    values = returns.dropna().astype(float)
    if not isinstance(values.index, pd.DatetimeIndex) or len(values) < 2:
        raise ValueError("report returns require at least two timestamped observations")
    equity = (1 + values).cumprod()
    monthly = (1 + values).groupby([values.index.year, values.index.month]).prod() - 1
    years = sorted(set(values.index.year))
    heatmap = np.full((len(years), 12), np.nan)
    year_row = {year: row for row, year in enumerate(years)}
    for (year, month), value in monthly.items():
        heatmap[year_row[year], month - 1] = value

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(np.arange(1, len(equity) + 1), equity, color="#38d39f", label="Observed")
    cone = monte_carlo.equity_cone
    axes[0, 0].fill_between(cone.index, cone["p05"], cone["p95"], color="#38d39f", alpha=0.12, label="MC 5–95%")
    axes[0, 0].plot(cone.index, cone["p50"], color="#87a49b", linewidth=1, label="MC median")
    axes[0, 0].set_title("Equity and Monte Carlo cone")
    axes[0, 0].legend()
    axes[0, 1].fill_between(values.index, drawdowns(values.to_numpy()), 0, color="#ff627d", alpha=0.7)
    axes[0, 1].set_title("Underwater curve")
    image = axes[1, 0].imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.2)
    axes[1, 0].set(xticks=range(12), xticklabels="Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), yticks=range(len(years)), yticklabels=years, title="Monthly returns")
    figure.colorbar(image, ax=axes[1, 0], format="%.0f%%")
    axes[1, 1].hist(values, bins=min(60, max(10, len(values) // 10)), color="#38d39f", alpha=0.75)
    axes[1, 1].axvline(0, color="#87a49b", linewidth=1)
    axes[1, 1].set_title("Return distribution")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, metadata={"Software": "tradingbot-bitso"})
    plt.close(figure)
    return destination
