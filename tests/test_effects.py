"""Tests for effect-size computation from raw study data.

Every expected value is derived from the closed-form definition
independently of the implementation.
"""

import math

import numpy as np
import pytest

from metanalysis import (
    effect_md,
    effect_smd,
    effect_or,
    effect_rr,
    effect_rd,
    effect_cor,
)


def test_mean_difference():
    # y = m1 - m2 ; v = sd1^2/n1 + sd2^2/n2
    yi, vi = effect_md(m1=[10, 5], sd1=[2, 1], n1=[50, 20],
                       m2=[8, 4], sd2=[3, 1], n2=[50, 25])
    assert yi[0] == pytest.approx(2.0)
    assert vi[0] == pytest.approx(4 / 50 + 9 / 50)
    assert yi[1] == pytest.approx(1.0)
    assert vi[1] == pytest.approx(1 / 20 + 1 / 25)


def test_standardized_mean_difference_hedges_g():
    yi, vi = effect_smd(m1=[10], sd1=[2], n1=[50], m2=[8], sd2=[3], n2=[50])
    sp = math.sqrt((49 * 4 + 49 * 9) / 98)          # 2.5495098
    d = 2.0 / sp                                     # 0.7844645
    J = 1 - 3 / (4 * 98 - 1)                          # 0.9923274
    vd = (50 + 50) / (50 * 50) + d ** 2 / (2 * (50 + 50))
    assert yi[0] == pytest.approx(J * d, rel=1e-9)
    assert vi[0] == pytest.approx(J ** 2 * vd, rel=1e-9)


def test_smd_without_small_sample_correction_is_cohens_d():
    yi, _ = effect_smd(m1=[10], sd1=[2], n1=[50], m2=[8], sd2=[3], n2=[50],
                       correct=False)
    sp = math.sqrt((49 * 4 + 49 * 9) / 98)
    assert yi[0] == pytest.approx(2.0 / sp, rel=1e-12)


def test_log_odds_ratio():
    # e1/n1 group 1 events/total ; e2/n2 group 2
    yi, vi = effect_or(e1=[15], n1=[100], e2=[10], n2=[100])
    a, b, c, d = 15, 85, 10, 90
    assert yi[0] == pytest.approx(math.log(a * d / (b * c)), rel=1e-12)
    assert vi[0] == pytest.approx(1 / a + 1 / b + 1 / c + 1 / d, rel=1e-12)


def test_log_odds_ratio_zero_cell_gets_continuity_correction():
    yi, vi = effect_or(e1=[0], n1=[50], e2=[5], n2=[50])
    a, b, c, d = 0.5, 50.5, 5.5, 45.5
    assert np.isfinite(yi[0])
    assert yi[0] == pytest.approx(math.log(a * d / (b * c)), rel=1e-12)
    assert vi[0] == pytest.approx(1 / a + 1 / b + 1 / c + 1 / d, rel=1e-12)


def test_log_risk_ratio():
    yi, vi = effect_rr(e1=[15], n1=[100], e2=[10], n2=[100])
    assert yi[0] == pytest.approx(math.log((15 / 100) / (10 / 100)), rel=1e-12)
    assert vi[0] == pytest.approx(1 / 15 - 1 / 100 + 1 / 10 - 1 / 100, rel=1e-12)


def test_risk_difference():
    yi, vi = effect_rd(e1=[15], n1=[100], e2=[10], n2=[100])
    assert yi[0] == pytest.approx(0.05, rel=1e-12)
    assert vi[0] == pytest.approx(15 * 85 / 100 ** 3 + 10 * 90 / 100 ** 3,
                                  rel=1e-12)


def test_fisher_z_correlation():
    yi, vi = effect_cor(r=[0.5], n=[28])
    assert yi[0] == pytest.approx(math.atanh(0.5), rel=1e-12)
    assert vi[0] == pytest.approx(1 / (28 - 3), rel=1e-12)


def test_effect_cor_requires_n_above_three():
    # Fisher-z variance 1/(n-3) is inf at n=3 and NEGATIVE at n<3.
    with pytest.raises(ValueError):
        effect_cor(r=[0.5], n=[3])
    with pytest.raises(ValueError):
        effect_cor(r=[0.5], n=[2])


def test_effect_cor_rejects_unit_or_out_of_range_correlation():
    # arctanh(+-1) = inf and arctanh(|r|>1) = nan.
    with pytest.raises(ValueError):
        effect_cor(r=[1.0], n=[30])
    with pytest.raises(ValueError):
        effect_cor(r=[-1.5], n=[30])


def test_effect_smd_requires_more_than_two_total():
    # n1 + n2 == 2 -> df = 0 -> pooled SD divides by zero -> (nan, nan).
    with pytest.raises(ValueError):
        effect_smd(m1=[10], sd1=[2], n1=[1], m2=[8], sd2=[3], n2=[1])


def test_effects_are_numpy_arrays():
    yi, vi = effect_md(m1=[1, 2], sd1=[1, 1], n1=[10, 10],
                       m2=[0, 0], sd2=[1, 1], n2=[10, 10])
    assert isinstance(yi, np.ndarray)
    assert isinstance(vi, np.ndarray)
