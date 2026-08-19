from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evaluate import drawdown


def _last_n(df: pd.DataFrame, n: int | None):
    return df if n is None else df.tail(n)


def plot_candles_with_indicators(df: pd.DataFrame, n: int = 500, title: str = "Candles + indicators"):
    d = _last_n(df, n)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=("Price", "RSI", "ATR / volatility")
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="OHLC"
    ), row=1, col=1)
    for col in ["ema20", "ema50", "ema200"]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=col), row=1, col=1)
    if "rsi14" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["rsi14"], mode="lines", name="RSI 14"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    if "atr" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["atr"], mode="lines", name="ATR"), row=3, col=1)
    if "atr_fast_slow" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["atr_fast_slow"], mode="lines", name="ATR fast/slow", yaxis="y4"), row=3, col=1)
    fig.update_layout(title=title, height=820, xaxis_rangeslider_visible=False, template="plotly_white")
    return fig


def plot_sessions_by_hour(df: pd.DataFrame, title="Average range and volume by hour"):
    d = df.copy()
    d["hour"] = d.index.hour
    d["range"] = d["High"] - d["Low"]
    hourly = d.groupby("hour").agg(avg_range=("range", "mean"), avg_volume=("Volume", "mean"), bars=("Close", "size")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["avg_range"], name="Avg high-low range"), secondary_y=False)
    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["avg_volume"], mode="lines+markers", name="Avg volume"), secondary_y=True)
    fig.update_layout(title=title, xaxis_title="Broker/server hour", height=430, template="plotly_white")
    fig.update_yaxes(title_text="Price range", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    return fig


def plot_feature_correlation(df: pd.DataFrame, feature_cols: list[str], title="Feature correlation heatmap"):
    corr = df[feature_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmid=0))
    fig.update_layout(title=title, height=760, template="plotly_white")
    return fig


def plot_trades_on_chart(df: pd.DataFrame, trades: pd.DataFrame, n: int = 500, title="Trades on chart"):
    d = _last_n(df, n)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    if trades is not None and not trades.empty:
        tr = trades.copy()
        tr = tr[(tr["entry_time"] >= d.index.min()) & (tr["entry_time"] <= d.index.max())]
        if not tr.empty:
            # Offset so triangles sit just outside the bar instead of on the price line.
            # 0.2 % of entry price gives ~4 pts on XAUUSD 2000 - visible but not intrusive.
            offset = tr["entry_price"] * 0.002

            longs  = tr[tr["direction"] ==  1]
            shorts = tr[tr["direction"] == -1]

            # -- Entry markers ------------------------------------------------
            # Long  : green  ▲  placed BELOW entry price
            if not longs.empty:
                fig.add_trace(go.Scatter(
                    x=longs["entry_time"],
                    y=longs["entry_price"] - offset[longs.index],
                    mode="markers", name="Long entry",
                    marker=dict(
                        symbol="triangle-up", size=13,
                        color="#00c853",
                        line=dict(color="#007c2e", width=1),
                    ),
                ))

            # Short : red    ▼  placed ABOVE entry price
            if not shorts.empty:
                fig.add_trace(go.Scatter(
                    x=shorts["entry_time"],
                    y=shorts["entry_price"] + offset[shorts.index],
                    mode="markers", name="Short entry",
                    marker=dict(
                        symbol="triangle-down", size=13,
                        color="#d50000",
                        line=dict(color="#7f0000", width=1),
                    ),
                ))

            # -- Exit markers split by reason ---------------------------------
            tp_exits    = tr[tr["exit_reason"] == "TP"]
            sl_exits    = tr[tr["exit_reason"] == "SL"]
            other_exits = tr[~tr["exit_reason"].isin(["TP", "SL"])]

            if not tp_exits.empty:
                fig.add_trace(go.Scatter(
                    x=tp_exits["exit_time"], y=tp_exits["exit_price"],
                    mode="markers", name="Exit - TP",
                    marker=dict(symbol="circle", size=9,
                                color="#00c853", line=dict(color="#007c2e", width=1)),
                ))
            if not sl_exits.empty:
                fig.add_trace(go.Scatter(
                    x=sl_exits["exit_time"], y=sl_exits["exit_price"],
                    mode="markers", name="Exit - SL",
                    marker=dict(symbol="x-thin", size=11,
                                color="#d50000", line=dict(color="#7f0000", width=2)),
                ))
            if not other_exits.empty:
                fig.add_trace(go.Scatter(
                    x=other_exits["exit_time"], y=other_exits["exit_price"],
                    mode="markers", name="Exit - manual",
                    marker=dict(symbol="diamond", size=9, color="#9e9e9e"),
                ))

            # -- Dashed connector arrows + TP/SL brackets (last 50 trades) ----
            for _, r in tr.tail(50).iterrows():
                entry_t = r["entry_time"]
                exit_t  = r["exit_time"]
                entry_p = float(r["entry_price"])
                exit_p  = float(r["exit_price"])
                span_t  = exit_t - entry_t   # pd.Timedelta
                span_p  = exit_p - entry_p

                # Dashed line body - entry -> exit
                fig.add_shape(
                    type="line",
                    x0=entry_t, y0=entry_p,
                    x1=exit_t,  y1=exit_p,
                    xref="x", yref="y",
                    line=dict(color="black", width=1, dash="dash"),
                )

                # Arrowhead only: annotation tail sits 90 % of the way from
                # entry -> exit (data coordinates) so only the tip is visible,
                # giving the dashed-arrow effect without overriding the dashed line.
                if abs(span_t.total_seconds()) > 1 or abs(span_p) > 1e-8:
                    fig.add_annotation(
                        x=exit_t,  y=exit_p,
                        ax=exit_t - span_t * 0.10,
                        ay=exit_p - span_p * 0.10,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        text="", showarrow=True,
                        arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                        arrowcolor="black",
                    )

                # TP / SL bracket lines
                fig.add_shape(
                    type="line",
                    x0=entry_t, x1=exit_t, y0=r["tp"], y1=r["tp"],
                    line=dict(color="rgba(0,180,0,0.55)", dash="dot", width=1),
                )
                fig.add_shape(
                    type="line",
                    x0=entry_t, x1=exit_t, y0=r["sl"], y1=r["sl"],
                    line=dict(color="rgba(213,0,0,0.55)", dash="dot", width=1),
                )

    fig.update_layout(
        title=title, height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_equity_and_drawdown(equity_df: pd.DataFrame, title="Equity and drawdown"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
    if equity_df.empty:
        fig.update_layout(title="No equity data")
        return fig
    eq = equity_df["equity"].astype(float)
    dd = drawdown(eq) * 100
    fig.add_trace(go.Scatter(x=equity_df.index, y=eq, mode="lines", name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=equity_df.index, y=dd, mode="lines", name="Drawdown %", fill="tozeroy"), row=2, col=1)
    fig.update_layout(title=title, height=650, template="plotly_white")
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def save_html(fig, path):
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def save_html_combined(figs, path, title="RL Trading - Pipeline Report"):
    """Merge several plotly figures into a single standalone HTML document.

    ``figs`` is an iterable of ``(name, figure)`` tuples.  plotly.js is
    included once via CDN and each figure is embedded below an <h2> heading,
    so the result is one scrollable report instead of N separate files.
    """
    snippets = []
    for name, fig in figs:
        if fig is None:
            continue
        snippet = fig.to_html(full_html=False, include_plotlyjs=False)
        snippets.append(
            f'<h2 style="font-family:sans-serif;color:#1565c0;margin:40px 0 10px 0">{name}</h2>\n{snippet}'
        )
    body = "\n".join(snippets)
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        "</head>\n<body style=\"margin:24px;font-family:sans-serif;background:#fafafa\">\n"
        f"<h1 style=\"color:#0d47a1\">{title}</h1>\n{body}\n</body>\n</html>"
    )
    Path(path).write_text(html, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Training / validation analysis
# -----------------------------------------------------------------------------

def plot_insample_oos_equity(
    equities: dict[str, pd.DataFrame],
    initial_equity: float = 10_000.0,
    title: str = "Equity - in-sample -> out-of-sample",
) -> go.Figure:
    """Chain Train -> Val -> Test equity curves onto a single time axis.

    Each segment is rescaled so it starts exactly where the previous one ended,
    giving a single continuous equity narrative.  Vertical dashed lines mark
    the in-sample / out-of-sample boundaries with a <- label.

    Parameters
    ----------
    equities : ordered dict {"Train": eq_df, "Val": eq_df, "Test": eq_df}
               (Val may be omitted; any subset in chronological order is fine)
    initial_equity : starting equity that each raw curve begins from
    """
    palette = {
        "Train": ("#1565c0", "rgba(21,101,192,0.08)"),
        "Val":   ("#e65100", "rgba(230,81,0,0.08)"),
        "Test":  ("#2e7d32", "rgba(46,125,50,0.08)"),
    }

    fig = go.Figure()

    chain_end = initial_equity   # running equity at which the next segment anchors
    boundaries = []              # (timestamp, left_label, right_label)

    names = list(equities.keys())
    for idx, (name, eq_df) in enumerate(equities.items()):
        if eq_df is None or eq_df.empty:
            continue

        raw_eq = eq_df["equity"].astype(float)
        scale  = chain_end / initial_equity
        scaled = raw_eq * scale          # shift curve up/down to meet chain_end

        color, fill = palette.get(name, ("#607d8b", "rgba(96,125,139,0.06)"))

        # To make the curve visually connect at the boundary with the previous
        # segment, prepend the boundary point (time = start of this segment,
        # y = chain_end) only when there's a gap (embargo bars) between segments.
        x_vals = list(eq_df.index)
        y_vals = list(scaled)

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=fill,
        ))

        # Collect segment boundary for annotation (between this and next segment).
        if idx < len(names) - 1:
            boundaries.append((
                eq_df.index.max(),
                name,
                names[idx + 1],
            ))

        chain_end = float(scaled.iloc[-1])

    # Baseline (initial equity) horizontal reference.
    fig.add_hline(y=initial_equity, line_dash="dot",
                  line_color="gray", opacity=0.35)

    # Vertical separators at segment boundaries.
    for ts, left, right in boundaries:
        fig.add_vline(
            x=ts,
            line_dash="dash", line_color="#546e7a", line_width=1.5,
        )
        fig.add_annotation(
            x=ts, y=1.01, yref="paper",
            text=f"<- {left} | {right} ->",
            showarrow=False,
            font=dict(size=11, color="#546e7a"),
            xanchor="center",
        )

    fig.update_layout(
        title=title, height=520, template="plotly_white",
        yaxis_title="Equity ($, chained)",
        xaxis_title="Date",
        legend=dict(orientation="h", y=1.06, x=0),
    )
    return fig


def plot_equity_comparison(
    splits: dict[str, pd.DataFrame],
    title: str = "Equity curves - all splits (normalised to 100)",
) -> go.Figure:
    """Overlay equity curves for Train / Val / Test (and optional Baseline).

    Each curve is normalised to start at 100 so relative performance is
    visually comparable regardless of initial equity.
    """
    palette = {
        "Train": "#1976d2", "Val": "#f57c00",
        "Test": "#388e3c", "Baseline": "#9e9e9e",
    }
    fig = go.Figure()
    for name, eq_df in splits.items():
        if eq_df is None or eq_df.empty:
            continue
        eq = eq_df["equity"].astype(float)
        fig.add_trace(go.Scatter(
            x=eq_df.index, y=eq / eq.iloc[0] * 100,
            mode="lines", name=name,
            line=dict(color=palette.get(name, "#607d8b"), width=2),
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title=title, height=480, template="plotly_white",
        yaxis_title="Equity (start = 100)", xaxis_title="Date",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def plot_r_distribution(
    trades: pd.DataFrame,
    title: str = "Realized R-multiple distribution",
) -> go.Figure:
    """Histogram of realized R-multiples split by win (green) / loss (red)."""
    if trades is None or trades.empty:
        return go.Figure().update_layout(title="No trades")
    r = trades["r_mult"].astype(float).dropna()
    wins, losses = r[r > 0], r[r <= 0]
    bin_size = 0.25
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=losses, name="Loss",
                               marker_color="rgba(213,0,0,0.70)",
                               xbins=dict(size=bin_size)))
    fig.add_trace(go.Histogram(x=wins, name="Win",
                               marker_color="rgba(0,200,83,0.70)",
                               xbins=dict(size=bin_size)))
    fig.add_vline(x=0, line_color="black", line_width=1.5)
    if np.isfinite(r.mean()):
        fig.add_vline(x=r.mean(), line_dash="dash", line_color="#1976d2",
                      annotation_text=f"Mean {r.mean():.2f}R",
                      annotation_position="top right")
    win_rate   = (r > 0).mean() * 100
    expectancy = r.mean()
    fig.update_layout(
        title=(f"{title}   |   WR {win_rate:.1f}%   "
               f"Expectancy {expectancy:+.3f}R   n={len(r)}"),
        barmode="overlay", height=420, template="plotly_white",
        xaxis_title="Realized R-multiple", yaxis_title="Count",
    )
    return fig


def plot_monthly_pnl(
    trades: pd.DataFrame,
    title: str = "Monthly P&L (R-multiples)",
) -> go.Figure:
    """Bar chart of monthly total R with cumulative line on a secondary axis."""
    if trades is None or trades.empty:
        return go.Figure().update_layout(title="No trades")
    tr = trades.copy()
    tr["month"] = pd.to_datetime(tr["exit_time"]).dt.to_period("M").astype(str)
    monthly = (
        tr.groupby("month", as_index=False)["r_mult"]
        .sum()
        .rename(columns={"r_mult": "r_total"})
    )
    monthly["cumulative_r"] = monthly["r_total"].cumsum()
    colors = ["#00c853" if v >= 0 else "#d50000" for v in monthly["r_total"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["r_total"],
                         name="Monthly R", marker_color=colors), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["cumulative_r"],
                             mode="lines+markers", name="Cumulative R",
                             line=dict(color="#1976d2", width=2)), secondary_y=True)
    fig.add_hline(y=0, line_color="black", line_width=1, secondary_y=False)
    fig.update_layout(title=title, height=440, template="plotly_white")
    fig.update_yaxes(title_text="R per month", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative R", secondary_y=True)
    return fig


def plot_exit_breakdown(
    trades: pd.DataFrame,
    title: str = "Exit reason breakdown by trade direction",
) -> go.Figure:
    """Stacked bar: TP / SL / manual exits for Long vs Short."""
    if trades is None or trades.empty:
        return go.Figure().update_layout(title="No trades")
    tr = trades.copy()
    tr["side"] = tr["direction"].map({1: "Long", -1: "Short"})
    counts = tr.groupby(["side", "exit_reason"]).size().unstack(fill_value=0)
    reason_colors = {"TP": "#00c853", "SL": "#d50000",
                     "manual_close": "#9e9e9e", "flip_close": "#ff9800"}
    fig = go.Figure()
    for reason in counts.columns:
        fig.add_trace(go.Bar(x=counts.index, y=counts[reason], name=reason,
                             marker_color=reason_colors.get(reason, "#607d8b")))
    fig.update_layout(title=title, barmode="stack", height=400,
                      template="plotly_white", yaxis_title="Number of trades")
    return fig


def plot_entry_timing(
    trades: pd.DataFrame,
    title: str = "Trade entry - hour x weekday frequency",
) -> go.Figure:
    """Heat map: how many trades were entered at each (weekday, hour) cell."""
    if trades is None or trades.empty:
        return go.Figure().update_layout(title="No trades")
    tr = trades.copy()
    dt = pd.to_datetime(tr["entry_time"])
    tr["hour"] = dt.dt.hour
    tr["dow"]  = dt.dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    order = [d for d in order if d in tr["dow"].unique()]
    heat = tr.groupby(["dow", "hour"]).size().unstack(fill_value=0).reindex(order)
    fig = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
        colorscale="Blues", colorbar_title="# entries",
    ))
    fig.update_layout(title=title, height=360, template="plotly_white",
                      xaxis_title="Hour of day", yaxis_title="Day of week")
    return fig


def plot_bracket_usage(
    trades: pd.DataFrame,
    sl_atr_multipliers: tuple | list,
    tp_r_multipliers: tuple | list,
    title: str = "Agent bracket selection - SL x TP heatmap",
) -> go.Figure:
    """Heatmap showing how often the agent chose each (SL, TP) bracket,
    with the average realized R annotated in each cell.

    Requires 'sl_atr_mult' and 'tp_r_bracket' columns (added in the updated
    BracketTradingEnv._close_position).
    """
    if trades is None or trades.empty:
        return go.Figure().update_layout(title="No trades")
    if "sl_atr_mult" not in trades.columns or "tp_r_bracket" not in trades.columns:
        return go.Figure().update_layout(
            title="Bracket columns missing - re-run simulation with updated env")

    tr = trades.copy()
    # Snap to nearest grid value (float rounding safety).
    tr["sl_bin"] = tr["sl_atr_mult"].apply(
        lambda v: min(sl_atr_multipliers, key=lambda x: abs(x - v)))
    tr["tp_bin"] = tr["tp_r_bracket"].apply(
        lambda v: min(tp_r_multipliers, key=lambda x: abs(x - v)))

    sl_vals = sorted(sl_atr_multipliers)
    tp_vals = sorted(tp_r_multipliers)

    counts = (tr.groupby(["sl_bin", "tp_bin"]).size()
                .unstack(fill_value=0)
                .reindex(index=sl_vals, columns=tp_vals, fill_value=0))
    avg_r = (tr.groupby(["sl_bin", "tp_bin"])["r_mult"].mean()
               .unstack(fill_value=np.nan)
               .reindex(index=sl_vals, columns=tp_vals, fill_value=np.nan))

    cell_text = [
        [f"n={int(counts.iloc[i, j])}<br>avgR={avg_r.iloc[i, j]:+.2f}"
         if np.isfinite(avg_r.iloc[i, j]) else f"n={int(counts.iloc[i, j])}"
         for j in range(len(tp_vals))]
        for i in range(len(sl_vals))
    ]
    fig = go.Figure(go.Heatmap(
        z=counts.values,
        x=[f"TP {v}R" for v in tp_vals],
        y=[f"SL {v}xATR" for v in sl_vals],
        text=cell_text, texttemplate="%{text}",
        colorscale="Blues", colorbar_title="# trades",
    ))
    fig.update_layout(title=title, height=380, template="plotly_white",
                      xaxis_title="TP R-multiple", yaxis_title="SL ATR multiplier")
    return fig


def plot_rolling_metrics(
    equity_df: pd.DataFrame,
    window: int = 50,
    periods_per_year: int = 6_003,
    title: str | None = None,
) -> go.Figure:
    """Equity curve (top) + rolling Sharpe (bottom) over a sliding window."""
    if equity_df is None or equity_df.empty:
        return go.Figure().update_layout(title="No equity data")
    eq  = equity_df["equity"].astype(float)
    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan)
    roll_sharpe = (ret.rolling(window).mean()
                   / ret.rolling(window).std().replace(0, np.nan)
                   * np.sqrt(periods_per_year))
    title = title or f"Equity + rolling Sharpe  (window = {window} bars)"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.55, 0.45],
        subplot_titles=("Equity", f"Rolling Sharpe ({window}-bar window)"),
    )
    fig.add_trace(go.Scatter(x=equity_df.index, y=eq, mode="lines",
                             name="Equity", line=dict(color="#1976d2", width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=equity_df.index, y=roll_sharpe, mode="lines",
                             name="Rolling Sharpe", line=dict(color="#f57c00", width=1.5),
                             fill="tozeroy", fillcolor="rgba(245,124,0,0.12)"),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6, row=2, col=1)
    fig.update_layout(title=title, height=560, template="plotly_white", showlegend=False)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    return fig


def plot_metrics_table(
    metrics: dict[str, dict],
    title: str = "Performance comparison",
) -> go.Figure:
    """Plotly table comparing key metrics across named splits side-by-side.

    metrics = {"Train": {...}, "Val": {...}, "Test": {...}}
    Values come from evaluate.full_report().
    """
    keys = [
        "total_return_pct", "annualized_return_pct", "max_drawdown_pct",
        "sharpe_like", "sortino_ratio", "calmar_ratio",
        "profit_factor", "win_rate_pct", "avg_r", "n_trades",
    ]
    labels = [
        "Total Return %", "Annualised Return %", "Max Drawdown %",
        "Sharpe", "Sortino", "Calmar",
        "Profit Factor", "Win Rate %", "Avg R", "# Trades",
    ]
    splits = list(metrics.keys())

    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:.3f}" if isinstance(v, float) else str(int(v))

    # Build column lists for Plotly Table (one list per column).
    col_metric = labels
    cols_splits = [[_fmt(metrics[s].get(k)) for k in keys] for s in splits]

    # Cell background colours: green/red depending on sign / context.
    good = {"total_return_pct", "annualized_return_pct", "sharpe_like",
            "sortino_ratio", "calmar_ratio", "profit_factor", "win_rate_pct", "avg_r"}
    bad  = {"max_drawdown_pct"}

    def _cell_color(k, v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "#f5f5f5"
        if k in good:
            return "#c8e6c9" if v > 0 else "#ffcdd2"
        if k in bad:
            return "#ffcdd2" if v < -20 else "#f5f5f5"
        return "#f5f5f5"

    fill_metric_col = ["#f5f5f5"] * len(keys)
    fill_split_cols = [
        [_cell_color(k, metrics[s].get(k)) for k in keys]
        for s in splits
    ]

    fig = go.Figure(go.Table(
        columnwidth=[180] + [100] * len(splits),
        header=dict(
            values=["<b>Metric</b>"] + [f"<b>{s}</b>" for s in splits],
            fill_color="#1565c0",
            font=dict(color="white", size=13),
            align="left", height=34,
        ),
        cells=dict(
            values=[col_metric] + cols_splits,
            fill_color=[fill_metric_col] + fill_split_cols,
            align=["left"] + ["right"] * len(splits),
            font=dict(size=12), height=28,
        ),
    ))
    fig.update_layout(
        title=title,
        height=60 + len(keys) * 32,
        template="plotly_white",
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig
