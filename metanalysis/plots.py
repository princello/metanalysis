"""Forest and funnel plots for meta-analysis results.

Styling follows publication conventions: Arial/Helvetica sans-serif, thin
spines, color-blind-safe accents, and sizes chosen to stay legible at a
single-column display width.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .bias import egger_test
from .pooling import MetaResult

__all__ = ["forest_plot", "funnel_plot"]

MM = 1 / 25.4
_STUDY_C = "#0072B2"     # study points / CIs
_POOLED_C = "#252525"    # pooled diamond
_PI_C = "#D55E00"        # prediction interval
_REF_C = "#999999"       # reference / funnel lines


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.5, labelsize=7)


def _z(level):
    return float(stats.norm.ppf(0.5 + level / 2))


def forest_plot(result: MetaResult, labels=None, ax=None, xlabel=None,
                exp=False, show_pi=True, title=None):
    """Draw a forest plot from a :class:`MetaResult`.

    Each study is a marker (area proportional to its meta-analytic weight)
    with its confidence interval; the pooled estimate is a diamond at the
    bottom, optionally with a prediction interval whisker.

    Parameters
    ----------
    result : MetaResult
        Output of :func:`metanalysis.meta_analyze`.
    labels : sequence of str, optional
        Study labels (length must equal the number of studies).
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if omitted.
    xlabel : str, optional
        X-axis label (defaults to "Effect size" or "Effect (ratio scale)").
    exp : bool
        If True, exponentiate effects and use a log x-axis (for log OR/RR;
        the reference line is drawn at 1).
    show_pi : bool
        Draw the prediction interval under the diamond (random-effects only).
    title : str, optional
        Panel title. Defaults to a one-line summary of the pooled result.
    """
    import matplotlib.pyplot as plt

    yi = np.asarray(result.yi, dtype=float)
    sei = np.sqrt(result.vi)
    k = result.k
    if labels is None:
        labels = [f"Study {i + 1}" for i in range(k)]
    if len(labels) != k:
        raise ValueError(f"labels must have length {k}, got {len(labels)}")

    zc = _z(result.level)
    lo = yi - zc * sei
    hi = yi + zc * sei

    tx = (lambda v: np.exp(v)) if exp else (lambda v: v)
    ref = 1.0 if exp else 0.0

    if ax is None:
        height = max(55 * MM, (k + 3) * 9 * MM)
        _, ax = plt.subplots(figsize=(150 * MM, height))
    _apply_style(ax)

    # Rows: studies stacked from top, pooled result on the bottom row (y=0).
    y_studies = np.arange(k, 0, -1)
    # Marker area proportional to weight (weights are percentages summing ~100).
    w = np.asarray(result.weights, dtype=float)
    sizes = 20 + 180 * (w / w.max())

    ax.axvline(tx(ref), color=_REF_C, lw=0.8, ls="--", zorder=0)

    for yrow, l, h, y0, s in zip(y_studies, lo, hi, yi, sizes):
        ax.plot([tx(l), tx(h)], [yrow, yrow], color=_STUDY_C, lw=1.1, zorder=2)
        ax.scatter([tx(y0)], [yrow], s=s, marker="s", color=_STUDY_C,
                   zorder=3, edgecolors="white", linewidths=0.4)

    # Pooled diamond at y=0.
    est, cl, ch = result.estimate, result.ci_low, result.ci_high
    dx = [tx(cl), tx(est), tx(ch), tx(est)]
    dy = [0, 0.32, 0, -0.32]
    ax.fill(dx, dy, color=_POOLED_C, zorder=4)

    # Prediction interval whisker under the diamond.
    if show_pi and result.pi_low is not None:
        ax.plot([tx(result.pi_low), tx(result.pi_high)], [-0.55, -0.55],
                color=_PI_C, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([tx(result.pi_low)] * 2, [-0.7, -0.4], color=_PI_C, lw=1.0)
        ax.plot([tx(result.pi_high)] * 2, [-0.7, -0.4], color=_PI_C, lw=1.0)

    # Y ticks: study labels + a "Pooled" row.
    ax.set_yticks(list(y_studies) + [0])
    ax.set_yticklabels(list(labels) + ["Pooled"], fontsize=7)
    for t in ax.get_yticklabels()[-1:]:
        t.set_fontweight("bold")
    ax.set_ylim(-1.1, k + 0.8)

    # Right-margin annotation: estimate [CI] per study and for the pooled row.
    xr = ax.get_xlim()[1]
    for yrow, y0, l, h in zip(y_studies, yi, lo, hi):
        ax.text(1.02, yrow, f"{tx(y0):.2f} [{tx(l):.2f}, {tx(h):.2f}]",
                transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=6, color="#333333")
    ax.text(1.02, 0, f"{tx(est):.2f} [{tx(cl):.2f}, {tx(ch):.2f}]",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=6, fontweight="bold", color="#000000")

    if exp:
        ax.set_xscale("log")
    if xlabel is None:
        xlabel = "Effect (ratio scale)" if exp else "Effect size"
    ax.set_xlabel(xlabel, fontsize=8)

    if title is None:
        pct = int(round(result.level * 100))
        het = f"I²={100 * result.I2:.0f}%, τ²={result.tau2:.3f}"
        title = (f"Pooled {tx(est):.2f} "
                 f"[{tx(cl):.2f}, {tx(ch):.2f}] ({pct}% CI); {het}")
    ax.set_title(title, fontsize=8.5, loc="left", pad=6)

    ax.figure.subplots_adjust(left=0.24, right=0.74, top=0.9, bottom=0.12)
    return ax


def funnel_plot(result: MetaResult = None, yi=None, sei=None, ax=None,
                egger=True, level=0.95, title=None):
    """Draw a funnel plot with pseudo confidence-region lines.

    Studies are plotted as effect size (x) against standard error (y, inverted
    so precise studies sit at the top). Diagonal guide lines mark the expected
    ``level`` region under no small-study effects, centered on the pooled
    estimate. With ``egger=True`` the panel is annotated with Egger's test.

    Provide either a :class:`MetaResult` or raw ``yi``/``sei`` arrays.
    """
    import matplotlib.pyplot as plt

    if result is not None:
        yi = np.asarray(result.yi, dtype=float)
        sei = np.sqrt(result.vi)
        center = result.estimate
    else:
        if yi is None or sei is None:
            raise ValueError("provide a MetaResult, or both yi and sei")
        yi = np.asarray(yi, dtype=float)
        sei = np.asarray(sei, dtype=float)
        wf = 1.0 / sei ** 2
        center = float((wf * yi).sum() / wf.sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(100 * MM, 90 * MM))
    _apply_style(ax)

    se_max = float(sei.max())
    zc = _z(level)
    # Pseudo confidence-region funnel: center ± zc*se as se goes 0 -> se_max.
    se_line = np.array([0.0, se_max * 1.05])
    ax.plot(center + zc * se_line, se_line, color=_REF_C, lw=0.8, ls="--")
    ax.plot(center - zc * se_line, se_line, color=_REF_C, lw=0.8, ls="--")
    ax.axvline(center, color=_REF_C, lw=0.8, ls="-")

    ax.scatter(yi, sei, s=28, color=_STUDY_C, edgecolors="white",
               linewidths=0.4, zorder=3)

    ax.set_ylim(se_max * 1.1, 0)  # invert: precise studies on top
    ax.set_xlabel("Effect size", fontsize=8)
    ax.set_ylabel("Standard error", fontsize=8)

    if title is None:
        title = "Funnel plot"
    ax.set_title(title, fontsize=8.5, loc="left", pad=6)

    if egger and yi.size >= 3:
        eg = egger_test(yi=yi, sei=sei)
        txt = (f"Egger's test\nintercept = {eg.intercept:.2f}\n"
               f"p = {eg.pval:.3f}")
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, va="bottom",
                ha="left", fontsize=6, fontstyle="italic", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffce8",
                          edgecolor="#cccccc", linewidth=0.4))

    ax.figure.subplots_adjust(left=0.16, right=0.95, top=0.9, bottom=0.14)
    return ax
