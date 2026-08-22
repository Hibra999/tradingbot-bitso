from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from validation import MonteCarloResult, advanced_metrics, drawdowns

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_MISSING_ARIAL_MESSAGE = "findfont: Font family 'Arial' not found."


def _keep_matplotlib_log(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith(_MISSING_ARIAL_MESSAGE)


def generate_report(
    returns: pd.Series,
    monte_carlo: MonteCarloResult,
    destination: str | Path,
    *,
    agent_name: str = "Strategy",
    benchmark: pd.Series | None = None,
    comparators: dict[str, pd.Series] | None = None,
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
    axes[0, 0].plot(np.arange(1, len(equity) + 1), equity, color="#38d39f", label=agent_name)
    if benchmark is not None:
        benchmark_equity = (1 + benchmark).cumprod()
        axes[0, 0].plot(
            np.arange(1, len(benchmark_equity) + 1),
            benchmark_equity,
            color="#f6c85f",
            linewidth=1.2,
            label="Buy & Hold",
        )
    for name, comparator in (comparators or {}).items():
        comparator_equity = (1 + comparator).cumprod()
        axes[0, 0].plot(
            np.arange(1, len(comparator_equity) + 1),
            comparator_equity,
            linewidth=1.1,
            label=name,
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


def _quantstats_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    if result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    return result


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


def _telegram_table(
    title: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    widths = [max(len(row[i]) for row in (headers, *rows)) for i in range(len(headers))]
    lines = [
        "  ".join(
            value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
            for i, value in enumerate(row)
        ).rstrip()
        for row in (headers, *rows)
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return f"<b>{html.escape(title)}</b>\n<pre>{html.escape(chr(10).join(lines))}</pre>"


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
    agent_name: str = "Strategy",
    benchmark: pd.Series | None = None,
    comparators: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    values = returns.dropna().astype(float)
    if not isinstance(values.index, pd.DatetimeIndex) or len(values) < 2:
        raise ValueError("report returns require at least two timestamped observations")
    benchmark_values: pd.Series | None = None
    if benchmark is not None:
        candidate = benchmark.dropna().astype(float).rename("Buy & Hold")
        if not isinstance(candidate.index, pd.DatetimeIndex):
            raise ValueError("report benchmark requires timestamped observations")
        aligned = pd.concat((values.rename(agent_name), candidate), axis=1, join="inner").dropna()
        if len(aligned) < 2:
            raise ValueError("report strategy and benchmark require at least two aligned observations")
        values, benchmark_values = aligned[agent_name], aligned["Buy & Hold"]
    comparator_values: dict[str, pd.Series] = {}
    if comparators:
        candidates = {
            name: series.dropna().astype(float).rename(name)
            for name, series in comparators.items()
        }
        if any(not isinstance(series.index, pd.DatetimeIndex) for series in candidates.values()):
            raise ValueError("report comparators require timestamped observations")
        columns = [values.rename(agent_name)]
        if benchmark_values is not None:
            columns.append(benchmark_values.rename("Buy & Hold"))
        columns.extend(candidates.values())
        aligned = pd.concat(columns, axis=1, join="inner").dropna()
        if len(aligned) < 2:
            raise ValueError("report strategies require at least two aligned observations")
        values = aligned[agent_name]
        if benchmark_values is not None:
            benchmark_values = aligned["Buy & Hold"]
        comparator_values = {name: aligned[name] for name in candidates}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    graphics = destination.with_suffix(".png")
    text_path = destination.with_suffix(".txt")
    destination.with_suffix(".tex").unlink(missing_ok=True)
    generate_report(
        values,
        monte_carlo,
        graphics,
        agent_name=agent_name,
        benchmark=benchmark_values,
        comparators=comparator_values,
    )
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

    report_rows = [report_row(agent_name, values)]
    benchmark_metrics: dict[str, float] | None = None
    benchmark_total: float | None = None
    if benchmark_values is not None:
        report_rows.append(report_row("Buy & Hold", benchmark_values))
        benchmark_array = benchmark_values.to_numpy(dtype=float)
        benchmark_metrics = advanced_metrics(benchmark_array, periods)
        benchmark_total = float(np.prod(1 + benchmark_array) - 1)
    for name, comparator in comparator_values.items():
        report_rows.append(report_row(name, comparator))
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
    metric_headers = ("Metric", agent_name, "Buy & Hold") if benchmark_metrics else ("Metric", agent_name)
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
    comparator_metrics: dict[str, dict[str, float]] = {}
    if comparator_values:
        names = list(comparator_values)
        displays: dict[str, dict[str, str]] = {}
        for name, comparator in comparator_values.items():
            data = comparator.to_numpy(dtype=float)
            current_metrics = advanced_metrics(data, periods)
            comparator_metrics[name] = current_metrics
            displays[name] = {
                "Backtesting from": str(comparator.index[0]),
                "Backtesting to": str(comparator.index[-1]),
                "Observations": str(len(comparator)),
                "Total return": _value(float(np.prod(1 + data) - 1), percent=True),
                "Sharpe": _value(current_metrics["sharpe"]),
                "Sortino": _value(current_metrics["sortino"]),
                "Calmar": _value(current_metrics["calmar"]),
                "SQN": _value(current_metrics["sqn"]),
                "Profit factor": _value(current_metrics["profit_factor"]),
                "Expectancy": _value(current_metrics["expectancy"], percent=True),
                "Expectancy ratio": _value(current_metrics["expectancy_ratio"]),
                "Win rate": _value(current_metrics["win_rate"], percent=True),
                "Maximum drawdown": _value(current_metrics["max_drawdown"], percent=True),
                "Average drawdown": _value(current_metrics["average_drawdown"], percent=True),
                "Maximum drawdown duration": f"{current_metrics['drawdown_duration_max']:.0f} periods",
                "Average drawdown duration": f"{current_metrics['drawdown_duration_mean']:.2f} periods",
                "Monte Carlo ruin 20%": "-",
                "Monte Carlo ruin 30%": "-",
            }
        comparison_rows = [
            (*row, *(displays[name][row[0]] for name in names)) for row in comparison_rows
        ]
        metric_headers = (*metric_headers, *names)
    compact_labels = {
        "Total return": "Return",
        "Sharpe": "Sharpe",
        "Sortino": "Sortino",
        "Calmar": "Calmar",
        "Profit factor": "Profit factor",
        "Win rate": "Win rate",
        "Maximum drawdown": "Max drawdown",
        "Monte Carlo ruin 20%": "MC ruin 20%",
    }
    compact_rows = [
        (compact_labels[row[0]], *row[1:])
        for row in comparison_rows
        if row[0] in compact_labels
    ]
    compact_headers = tuple(
        "B&H" if name == "Buy & Hold" else name
        for name in metric_headers
    )
    telegram_report = _telegram_table(title, compact_headers, compact_rows)
    text_report = "\n\n".join(
        (
            _ascii_table("BACKTESTING REPORT", report_headers, report_rows),
            _ascii_table("SUMMARY METRICS", metric_headers, comparison_rows),
        )
    )
    text_path.write_text(text_report + "\n", encoding="utf-8")
    plt.style.use("default")
    quantstats_values = _quantstats_series(values)
    quantstats_benchmark = (
        _quantstats_series(benchmark_values) if benchmark_values is not None else None
    )
    quantstats_error: str | None = None
    font_logger = logging.getLogger("matplotlib.font_manager")
    font_logger.addFilter(_keep_matplotlib_log)
    try:
        with plt.rc_context():
            import quantstats as qs

            plt.rcParams["font.family"] = "DejaVu Sans"
            qs.reports.html(
                quantstats_values,
                benchmark=quantstats_benchmark,
                output=str(destination),
                title=title,
                periods_per_year=periods,
                download_filename=destination.name,
            )
    except np.linalg.LinAlgError:
        quantstats_error = "QuantStats plots unavailable: return covariance is singular."
        plt.close("all")
        destination.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title></head><body>"
            f"<section class=\"local-report\"><h1>{html.escape(title)}</h1>"
            f"<p>{html.escape(quantstats_error)}</p></section></body></html>",
            encoding="utf-8",
        )
    finally:
        font_logger.removeFilter(_keep_matplotlib_log)
    encoded = base64.b64encode(graphics.read_bytes()).decode("ascii")
    extra = (
        "<style>.local-report{max-width:960px;margin:36px auto}.local-report table{width:100%;border-collapse:collapse}"
        ".local-report th,.local-report td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}"
        ".local-graphic{display:block;max-width:960px;width:100%;margin:24px auto}</style>"
        f"<section class=\"local-report\"><h2>Existing Pipeline Graphics</h2><img class=\"local-graphic\" src=\"data:image/png;base64,{encoded}\" alt=\"Pipeline graphics\"></section>"
        + _html_table("Backtesting Report", report_headers, report_rows)
        + _html_table("Summary Metrics", metric_headers, comparison_rows)
    )
    report = destination.read_text(encoding="utf-8")
    report = report.replace("</body>", extra + "</body>") if "</body>" in report else report + extra
    destination.write_text(report, encoding="utf-8")
    return {
        "html": destination,
        "graphics": graphics,
        "text": text_path,
        "text_report": text_report,
        "telegram_report": telegram_report,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "comparator_metrics": comparator_metrics,
        "quantstats_error": quantstats_error,
    }
