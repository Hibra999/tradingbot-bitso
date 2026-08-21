from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from validation import MonteCarloResult, advanced_metrics, drawdowns

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def generate_report(
    returns: pd.Series,
    monte_carlo: MonteCarloResult,
    destination: str | Path,
    *,
    benchmark: pd.Series | None = None,
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
    axes[0, 0].plot(np.arange(1, len(equity) + 1), equity, color="#38d39f", label="RL model")
    if benchmark is not None:
        benchmark_equity = (1 + benchmark).cumprod()
        axes[0, 0].plot(
            np.arange(1, len(benchmark_equity) + 1),
            benchmark_equity,
            color="#f6c85f",
            linewidth=1.2,
            label="Buy & Hold",
        )
    cone = monte_carlo.equity_cone
    axes[0, 0].fill_between(cone.index, cone["p05"], cone["p95"], color="#38d39f", alpha=0.12, label="MC 5-95%")
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


def _periods_per_year(index: pd.DatetimeIndex) -> int:
    seconds = float(index.to_series().diff().dropna().dt.total_seconds().median())
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("report timestamps must be strictly increasing")
    return max(1, round(365 * 24 * 60 * 60 / seconds))


def _value(value: float, *, percent: bool = False) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def _ascii_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    render = lambda row: "| " + " | ".join(
        value.ljust(widths[i]) if i == 0 else value.rjust(widths[i]) for i, value in enumerate(row)
    ) + " |"
    return "\n".join((title.center(len(border)), border, render(headers), border, *(render(row) for row in rows), border))


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    body = "\n".join(" & ".join(_tex(value) for value in row) + r" \\" for row in (headers, *rows))
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{_tex(title)}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'r' * (len(headers) - 1)}}}\n\\hline\n"
        f"{body}\n\\hline\n\\end{{tabular}}\n\\end{{table}}"
    )


def _html_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<section class=\"local-report\"><h2>{html.escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>"


def generate_full_report(
    returns: pd.Series,
    monte_carlo: MonteCarloResult,
    destination: str | Path,
    *,
    title: str = "Backtesting Report",
    symbol: str = "TOTAL",
    benchmark: pd.Series | None = None,
) -> dict[str, Any]:
    values = returns.dropna().astype(float)
    if not isinstance(values.index, pd.DatetimeIndex) or len(values) < 2:
        raise ValueError("report returns require at least two timestamped observations")
    benchmark_values: pd.Series | None = None
    if benchmark is not None:
        candidate = benchmark.dropna().astype(float).rename("Buy & Hold")
        if not isinstance(candidate.index, pd.DatetimeIndex):
            raise ValueError("report benchmark requires timestamped observations")
        aligned = pd.concat((values.rename("RL model"), candidate), axis=1, join="inner").dropna()
        if len(aligned) < 2:
            raise ValueError("report strategy and benchmark require at least two aligned observations")
        values, benchmark_values = aligned["RL model"], aligned["Buy & Hold"]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    graphics = destination.with_suffix(".png")
    text_path, latex_path = destination.with_suffix(".txt"), destination.with_suffix(".tex")
    generate_report(values, monte_carlo, graphics, benchmark=benchmark_values)
    periods = _periods_per_year(values.index)
    array = values.to_numpy()
    metrics = advanced_metrics(array, periods)
    total = float(np.prod(1 + array) - 1)
    report_headers = ("Strategy", "Pair", "Periods", "Avg Return %", "Tot Return %", "Win", "Draw", "Loss", "Win%")

    def report_row(name: str, series: pd.Series) -> tuple[str, ...]:
        data = series.to_numpy(dtype=float)
        strategy_wins = int(np.count_nonzero(data > 0))
        strategy_draws = int(np.count_nonzero(data == 0))
        strategy_losses = int(np.count_nonzero(data < 0))
        strategy_total = float(np.prod(1 + data) - 1)
        return (
            name,
            symbol,
            str(len(series)),
            f"{series.mean() * 100:.4f}",
            f"{strategy_total * 100:.4f}",
            str(strategy_wins),
            str(strategy_draws),
            str(strategy_losses),
            f"{strategy_wins / len(series) * 100:.2f}",
        )

    report_rows = [report_row("RL model", values)]
    benchmark_metrics: dict[str, float] | None = None
    benchmark_total: float | None = None
    if benchmark_values is not None:
        report_rows.append(report_row("Buy & Hold", benchmark_values))
        benchmark_array = benchmark_values.to_numpy(dtype=float)
        benchmark_metrics = advanced_metrics(benchmark_array, periods)
        benchmark_total = float(np.prod(1 + benchmark_array) - 1)
    metric_rows = [
        ("Backtesting from", str(values.index[0])),
        ("Backtesting to", str(values.index[-1])),
        ("Observations", str(len(values))),
        ("Total return", _value(total, percent=True)),
        ("Sharpe", _value(metrics["sharpe"])),
        ("Sortino", _value(metrics["sortino"])),
        ("Calmar", _value(metrics["calmar"])),
        ("SQN", _value(metrics["sqn"])),
        ("Profit factor", _value(metrics["profit_factor"])),
        ("Expectancy", _value(metrics["expectancy"], percent=True)),
        ("Expectancy ratio", _value(metrics["expectancy_ratio"])),
        ("Win rate", _value(metrics["win_rate"], percent=True)),
        ("Maximum drawdown", _value(metrics["max_drawdown"], percent=True)),
        ("Average drawdown", _value(metrics["average_drawdown"], percent=True)),
        ("Maximum drawdown duration", f"{metrics['drawdown_duration_max']:.0f} periods"),
        ("Average drawdown duration", f"{metrics['drawdown_duration_mean']:.2f} periods"),
        ("Monte Carlo ruin 20%", _value(monte_carlo.ruin_probability_20, percent=True)),
        ("Monte Carlo ruin 30%", _value(monte_carlo.ruin_probability_30, percent=True)),
    ]
    metric_headers = ("Metric", "RL model", "Buy & Hold") if benchmark_metrics else ("Metric", "RL model")
    comparison_rows = metric_rows
    if benchmark_metrics and benchmark_total is not None and benchmark_values is not None:
        benchmark_display = {
            "Backtesting from": str(benchmark_values.index[0]),
            "Backtesting to": str(benchmark_values.index[-1]),
            "Observations": str(len(benchmark_values)),
            "Total return": _value(benchmark_total, percent=True),
            "Sharpe": _value(benchmark_metrics["sharpe"]),
            "Sortino": _value(benchmark_metrics["sortino"]),
            "Calmar": _value(benchmark_metrics["calmar"]),
            "SQN": _value(benchmark_metrics["sqn"]),
            "Profit factor": _value(benchmark_metrics["profit_factor"]),
            "Expectancy": _value(benchmark_metrics["expectancy"], percent=True),
            "Expectancy ratio": _value(benchmark_metrics["expectancy_ratio"]),
            "Win rate": _value(benchmark_metrics["win_rate"], percent=True),
            "Maximum drawdown": _value(benchmark_metrics["max_drawdown"], percent=True),
            "Average drawdown": _value(benchmark_metrics["average_drawdown"], percent=True),
            "Maximum drawdown duration": f"{benchmark_metrics['drawdown_duration_max']:.0f} periods",
            "Average drawdown duration": f"{benchmark_metrics['drawdown_duration_mean']:.2f} periods",
            "Monte Carlo ruin 20%": "-",
            "Monte Carlo ruin 30%": "-",
        }
        comparison_rows = [(name, strategy, benchmark_display[name]) for name, strategy in metric_rows]
    text_report = "\n\n".join(
        (
            _ascii_table("BACKTESTING REPORT", report_headers, report_rows),
            _ascii_table("SUMMARY METRICS", metric_headers, comparison_rows),
        )
    )
    latex = "\n\n".join(
        (
            _latex_table("Backtesting Report", report_headers, report_rows),
            _latex_table("Summary Metrics", metric_headers, comparison_rows),
        )
    )
    text_path.write_text(text_report + "\n", encoding="utf-8")
    latex_path.write_text(latex + "\n", encoding="utf-8")
    import quantstats as qs

    plt.style.use("default")
    qs.reports.html(
        values,
        benchmark=benchmark_values,
        output=str(destination),
        title=title,
        periods_per_year=periods,
        download_filename=destination.name,
    )
    encoded = base64.b64encode(graphics.read_bytes()).decode("ascii")
    extra = (
        "<style>.local-report{max-width:960px;margin:36px auto}.local-report table{width:100%;border-collapse:collapse}"
        ".local-report th,.local-report td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}"
        ".local-graphic{display:block;max-width:960px;width:100%;margin:24px auto}</style>"
        f"<section class=\"local-report\"><h2>Existing Pipeline Graphics</h2><img class=\"local-graphic\" src=\"data:image/png;base64,{encoded}\" alt=\"Pipeline graphics\"></section>"
        + _html_table("Backtesting Report", report_headers, report_rows)
        + _html_table("Summary Metrics", metric_headers, comparison_rows)
        + f"<section class=\"local-report\"><h2>LaTeX Tables</h2><pre>{html.escape(latex)}</pre></section>"
    )
    report = destination.read_text(encoding="utf-8")
    report = report.replace("</body>", extra + "</body>") if "</body>" in report else report + extra
    destination.write_text(report, encoding="utf-8")
    return {
        "html": destination,
        "graphics": graphics,
        "latex": latex_path,
        "text": text_path,
        "text_report": text_report,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
    }
