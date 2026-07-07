"""Tests for forest and funnel plots.

Plots are validated behaviorally: the functions must return matplotlib Axes,
write non-empty image files, and validate their inputs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest

from metanalysis import meta_analyze, forest_plot, funnel_plot

YI = [0.1, 0.3, 0.5, -0.05, 0.22]
SEI = [0.10, 0.10, 0.12, 0.20, 0.08]


def _result():
    return meta_analyze(yi=YI, sei=SEI, method="DL")


def test_forest_plot_returns_axes():
    ax = forest_plot(_result())
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_forest_plot_writes_file(tmp_path):
    ax = forest_plot(_result(), labels=["A", "B", "C", "D", "E"])
    out = tmp_path / "forest.png"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


def test_forest_plot_rejects_wrong_number_of_labels():
    with pytest.raises(ValueError):
        forest_plot(_result(), labels=["only", "two"])


def test_funnel_plot_returns_axes():
    ax = funnel_plot(_result())
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_funnel_plot_writes_file(tmp_path):
    ax = funnel_plot(_result(), egger=True)
    out = tmp_path / "funnel.png"
    ax.figure.savefig(out)
    plt.close(ax.figure)
    assert out.exists() and out.stat().st_size > 0


def test_funnel_plot_accepts_raw_arrays():
    ax = funnel_plot(yi=YI, sei=SEI, egger=False)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_funnel_plot_rejects_invalid_level():
    with pytest.raises(ValueError):
        funnel_plot(yi=YI, sei=SEI, egger=False, level=1.5)
