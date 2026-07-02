"""Tests for core pooling against hand-computed reference values.

Reference dataset (chosen so every quantity is exactly hand-computable):
    yi  = [0.1, 0.3, 0.5]
    sei = [0.1, 0.1, 0.1]   -> vi = 0.01, wi = 100 each

Fixed effect:
    estimate = mean = 0.30
    se       = sqrt(1/300) = 0.05773503
Heterogeneity:
    Q   = 100*(0.04 + 0 + 0.04) = 8.0 ,  df = 2
    Qp  = chi2.sf(8, 2) = exp(-4) = 0.01831564
    I2  = (8-2)/8 = 0.75
    H2  = 8/2 = 4.0
DerSimonian-Laird:
    C     = 300 - 30000/300 = 200
    tau2  = (8-2)/200 = 0.03
    w*    = 1/(0.01+0.03) = 25 each ,  Sum = 75
    est   = 0.30 ,  se = sqrt(1/75) = 0.11547005
Prediction interval (k=3 -> t with 1 df, t.975 = 12.7062047):
    half  = 12.7062047 * sqrt(0.03 + 1/75) = 12.7062047 * 0.20816660 = 2.6449937
"""

import math

import numpy as np
import pytest

from metanalysis import meta_analyze

YI = [0.1, 0.3, 0.5]
SEI = [0.1, 0.1, 0.1]


def test_fixed_effect_estimate_and_se():
    res = meta_analyze(yi=YI, sei=SEI, method="FE")
    assert res.estimate == pytest.approx(0.30, abs=1e-12)
    assert res.se == pytest.approx(math.sqrt(1 / 300), rel=1e-12)


def test_fixed_effect_confidence_interval():
    res = meta_analyze(yi=YI, sei=SEI, method="FE")
    half = 1.959963985 * math.sqrt(1 / 300)
    assert res.ci_low == pytest.approx(0.30 - half, rel=1e-9)
    assert res.ci_high == pytest.approx(0.30 + half, rel=1e-9)


def test_heterogeneity_statistics():
    res = meta_analyze(yi=YI, sei=SEI, method="FE")
    assert res.Q == pytest.approx(8.0, rel=1e-12)
    assert res.Q_df == 2
    assert res.Q_pval == pytest.approx(math.exp(-4), rel=1e-9)
    assert res.I2 == pytest.approx(0.75, rel=1e-12)
    assert res.H2 == pytest.approx(4.0, rel=1e-12)


def test_fixed_effect_reports_zero_tau2():
    res = meta_analyze(yi=YI, sei=SEI, method="FE")
    assert res.tau2 == 0.0


def test_dersimonian_laird_tau2():
    res = meta_analyze(yi=YI, sei=SEI, method="DL")
    assert res.tau2 == pytest.approx(0.03, rel=1e-12)
    assert res.tau == pytest.approx(math.sqrt(0.03), rel=1e-12)


def test_dersimonian_laird_estimate_and_se():
    res = meta_analyze(yi=YI, sei=SEI, method="DL")
    assert res.estimate == pytest.approx(0.30, abs=1e-12)
    assert res.se == pytest.approx(math.sqrt(1 / 75), rel=1e-12)


def test_random_effects_prediction_interval():
    res = meta_analyze(yi=YI, sei=SEI, method="DL")
    half = 12.7062047 * math.sqrt(0.03 + 1 / 75)
    assert res.pi_low == pytest.approx(0.30 - half, rel=1e-6)
    assert res.pi_high == pytest.approx(0.30 + half, rel=1e-6)


def test_fixed_effect_has_no_prediction_interval():
    res = meta_analyze(yi=YI, sei=SEI, method="FE")
    assert res.pi_low is None
    assert res.pi_high is None


def test_reml_matches_dl_on_balanced_symmetric_data():
    # For this balanced, equal-variance, symmetric dataset REML and DL both give 0.03.
    res = meta_analyze(yi=YI, sei=SEI, method="REML")
    assert res.tau2 == pytest.approx(0.03, rel=1e-4)


def test_accepts_variances_instead_of_ses():
    vi = [0.01, 0.01, 0.01]
    res = meta_analyze(yi=YI, vi=vi, method="FE")
    assert res.estimate == pytest.approx(0.30, abs=1e-12)
    assert res.se == pytest.approx(math.sqrt(1 / 300), rel=1e-12)


def test_k_reports_number_of_studies():
    res = meta_analyze(yi=YI, sei=SEI, method="DL")
    assert res.k == 3


def test_single_study_has_exactly_zero_heterogeneity():
    # With one study heterogeneity is undefined; report clean zeros, no PI.
    for method in ("FE", "DL", "REML"):
        res = meta_analyze(yi=[0.4], sei=[0.1], method=method)
        assert res.tau2 == 0.0
        assert res.I2 == 0.0
        assert res.estimate == pytest.approx(0.4)
        assert res.pi_low is None
