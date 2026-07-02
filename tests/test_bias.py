"""Tests for Egger's regression test for funnel-plot asymmetry.

Egger's test regresses the standard normal deviate (yi/sei) on precision
(1/sei) by ordinary least squares; the intercept measures small-study
asymmetry and is tested with a t distribution on k-2 df.

Hand-computed reference:
    yi  = [0.2, 0.3, 0.4]
    sei = [0.2, 0.1, 0.05]
    SND  = [1, 3, 8] ,  prec = [5, 10, 20]
    slope     b = 55 / 116.6667 = 0.4714286
    intercept a = 4 - b*11.6667 = -1.5
    SSE = 0.0714286 , df = 1 , MSE = 0.0714286
    SE(a) = sqrt(MSE * (1/3 + mean_prec^2/Sxx)) = sqrt(0.0714286*1.5) = 0.327327
    t = -1.5 / 0.327327 = -4.58258
"""

import math

import pytest
from scipy import stats

from metanalysis import egger_test

YI = [0.2, 0.3, 0.4]
SEI = [0.2, 0.1, 0.05]


def test_egger_intercept():
    res = egger_test(yi=YI, sei=SEI)
    assert res.intercept == pytest.approx(-1.5, rel=1e-9)


def test_egger_intercept_se_and_t():
    res = egger_test(yi=YI, sei=SEI)
    assert res.se == pytest.approx(0.327327, rel=1e-5)
    assert res.t == pytest.approx(-4.58258, rel=1e-5)


def test_egger_degrees_of_freedom():
    res = egger_test(yi=YI, sei=SEI)
    assert res.df == 1


def test_egger_pvalue_matches_t_distribution():
    res = egger_test(yi=YI, sei=SEI)
    expected = 2 * stats.t.sf(abs(res.t), res.df)
    assert res.pval == pytest.approx(expected, rel=1e-9)


def test_egger_slope():
    res = egger_test(yi=YI, sei=SEI)
    assert res.slope == pytest.approx(55 / (350 / 3), rel=1e-9)


def test_egger_symmetric_data_has_intercept_near_zero():
    # Effects independent of precision -> no small-study effect.
    yi = [0.30, 0.30, 0.30, 0.30]
    sei = [0.30, 0.20, 0.10, 0.05]
    res = egger_test(yi=yi, sei=sei)
    assert res.intercept == pytest.approx(0.0, abs=1e-9)
    assert res.pval > 0.5


def test_egger_requires_at_least_three_studies():
    with pytest.raises(ValueError):
        egger_test(yi=[0.2, 0.3], sei=[0.1, 0.1])


def test_egger_accepts_variances():
    res_se = egger_test(yi=YI, sei=SEI)
    res_vi = egger_test(yi=YI, vi=[s ** 2 for s in SEI])
    assert res_vi.intercept == pytest.approx(res_se.intercept, rel=1e-12)
