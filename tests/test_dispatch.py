"""Tests for the compute_effects dispatcher and pandas/table input."""

import numpy as np
import pandas as pd
import pytest

from metanalysis import compute_effects, effect_or, effect_md, meta_analyze


def test_compute_effects_dispatches_to_or():
    got = compute_effects("OR", e1=[15], n1=[100], e2=[10], n2=[100])
    exp = effect_or(e1=[15], n1=[100], e2=[10], n2=[100])
    assert got[0] == pytest.approx(exp[0])
    assert got[1] == pytest.approx(exp[1])


def test_compute_effects_dispatches_to_md():
    got = compute_effects("MD", m1=[10], sd1=[2], n1=[50],
                          m2=[8], sd2=[3], n2=[50])
    exp = effect_md(m1=[10], sd1=[2], n1=[50], m2=[8], sd2=[3], n2=[50])
    assert got[0] == pytest.approx(exp[0])
    assert got[1] == pytest.approx(exp[1])


def test_compute_effects_is_case_insensitive():
    a = compute_effects("or", e1=[15], n1=[100], e2=[10], n2=[100])
    b = compute_effects("OR", e1=[15], n1=[100], e2=[10], n2=[100])
    assert a[0] == pytest.approx(b[0])


def test_compute_effects_rejects_unknown_measure():
    with pytest.raises(ValueError):
        compute_effects("NOPE", e1=[1], n1=[2], e2=[1], n2=[2])


def test_effects_accept_pandas_columns():
    # DataFrame columns (pandas Series) must flow through unchanged.
    df = pd.DataFrame({"e1": [15, 8], "n1": [100, 90],
                       "e2": [10, 5], "n2": [100, 95]})
    yi, vi = compute_effects("OR", e1=df["e1"], n1=df["n1"],
                             e2=df["e2"], n2=df["n2"])
    assert isinstance(yi, np.ndarray)
    res = meta_analyze(yi=yi, vi=vi, method="DL")
    assert res.k == 2
