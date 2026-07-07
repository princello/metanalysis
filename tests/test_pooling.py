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

from metanalysis import effect_rr, meta_analyze

YI = [0.1, 0.3, 0.5]
SEI = [0.1, 0.1, 0.1]

# Colditz (1994) BCG vaccine trials (metafor's dat.bcg): tpos, tneg, cpos, cneg.
_BCG = np.array([
    [4, 119, 11, 128], [6, 300, 29, 274], [3, 228, 11, 209],
    [62, 13536, 248, 12619], [33, 5036, 47, 5761], [180, 1361, 372, 1079],
    [8, 2537, 10, 619], [505, 87886, 499, 87892], [29, 7470, 45, 7232],
    [17, 1699, 65, 1600], [186, 50448, 141, 27197], [5, 2493, 3, 2338],
    [27, 16886, 29, 17825],
], dtype=float)


def _bcg_effects():
    tpos, tneg, cpos, cneg = _BCG[:, 0], _BCG[:, 1], _BCG[:, 2], _BCG[:, 3]
    return effect_rr(e1=tpos, n1=tpos + tneg, e2=cpos, n2=cpos + cneg)


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


# --- Hartung-Knapp-Sidik-Jonkman (HKSJ / "knha") -------------------------
#
# Reference values are metafor 5.0.1 rma(yi, vi, method="REML", test="knha")
# on the BCG log-risk-ratio effects (computed once, hard-coded here):
#   estimate = -0.7145323483651009
#   se       =  0.1807917455191005
#   ci.lb    = -1.1084437230000497
#   ci.ub    = -0.3206209737301521
#   tval     = -3.9522398896779904
#   pval     =  0.0019200150851340
#   dfs      = 12  (= k - 1)
# metafor's tau2 differs from ours by ~7e-8 (fixed-point vs. its solver), so
# the derived quantities agree to ~1e-8; we pin at abs=1e-6.


def test_knha_ci_matches_metafor_on_bcg():
    yi, vi = _bcg_effects()
    res = meta_analyze(yi=yi, vi=vi, method="REML", test="knha")
    assert res.test == "knha"
    assert res.estimate == pytest.approx(-0.7145323483651009, abs=1e-6)
    assert res.se == pytest.approx(0.1807917455191005, abs=1e-6)
    assert res.ci_low == pytest.approx(-1.1084437230000497, abs=1e-6)
    assert res.ci_high == pytest.approx(-0.3206209737301521, abs=1e-6)
    assert res.z == pytest.approx(-3.9522398896779904, abs=1e-6)
    assert res.pval == pytest.approx(0.0019200150851340, abs=1e-6)


def test_knha_se_is_wald_se_scaled_by_sqrt_q():
    # Internal consistency: se_knha = se_wald * sqrt(q) and the test df is k-1.
    yi, vi = _bcg_effects()
    ref = meta_analyze(yi=yi, vi=vi, method="REML")           # test="z"
    res = meta_analyze(yi=yi, vi=vi, method="REML", test="knha")

    tau2 = res.tau2
    w = 1.0 / (np.asarray(vi) + tau2)
    mu = (w * np.asarray(yi)).sum() / w.sum()
    q = (w * (np.asarray(yi) - mu) ** 2).sum() / (res.k - 1)

    assert res.se == pytest.approx(ref.se * math.sqrt(q), rel=1e-12)
    # t critical value with k-1 df is used for both CI and effect test.
    from scipy import stats
    tcrit = float(stats.t.ppf(0.975, df=res.k - 1))
    assert res.ci_high - res.estimate == pytest.approx(tcrit * res.se, rel=1e-12)
    assert res.z == pytest.approx(res.estimate / res.se, rel=1e-12)


def test_knha_leaves_point_estimate_tau2_and_q_unchanged():
    yi, vi = _bcg_effects()
    z = meta_analyze(yi=yi, vi=vi, method="REML")
    hk = meta_analyze(yi=yi, vi=vi, method="REML", test="knha")
    assert hk.estimate == z.estimate
    assert hk.tau2 == z.tau2
    assert hk.Q == z.Q
    assert hk.I2 == z.I2
    assert hk.pi_low == z.pi_low and hk.pi_high == z.pi_high


def test_default_test_is_z_and_unchanged_on_bcg():
    # Regression guard: default output must be byte-for-byte the pre-HKSJ result.
    yi, vi = _bcg_effects()
    res = meta_analyze(yi=yi, vi=vi, method="REML")
    assert res.test == "z"
    assert res.estimate == -0.714532342157376
    assert res.se == 0.17978151610327855
    assert res.ci_low == -1.0668976388058098
    assert res.ci_high == -0.3621670455089423
    assert res.z == -3.9744483061701446
    assert res.pval == 7.054258100504829e-05


def test_knha_works_for_dl():
    yi, vi = _bcg_effects()
    res = meta_analyze(yi=yi, vi=vi, method="DL", test="knha")
    assert res.test == "knha"
    # Wider than the naive z interval for these data (HKSJ inflates the SE).
    z = meta_analyze(yi=yi, vi=vi, method="DL")
    assert res.se > z.se


def test_knha_rejected_for_fixed_effect():
    yi, vi = _bcg_effects()
    with pytest.raises(ValueError):
        meta_analyze(yi=yi, vi=vi, method="FE", test="knha")


def test_knha_rejected_for_single_study():
    with pytest.raises(ValueError):
        meta_analyze(yi=[0.4], sei=[0.1], method="REML", test="knha")


def test_knha_rejected_when_heterogeneity_is_zero():
    # All effects identical -> residual mean square q = 0 -> HKSJ SE collapses
    # to 0. Must raise a clear error instead of crashing (ZeroDivisionError)
    # or fabricating p = 0.
    with pytest.raises(ValueError):
        meta_analyze(yi=[0.3, 0.3, 0.3], vi=[0.01, 0.02, 0.03],
                     method="DL", test="knha")


# ── Guards on degenerate / invalid inputs ──────────────────────────────
def test_nonfinite_variance_is_rejected():
    # NaN slips past `vi <= 0` (nan <= 0 is False); it must not silently
    # produce estimate = nan.
    with pytest.raises(ValueError):
        meta_analyze(yi=[0.1, 0.2], vi=[0.01, float("nan")], method="FE")
    with pytest.raises(ValueError):
        meta_analyze(yi=[0.1, 0.2], vi=[0.01, float("inf")], method="FE")


def test_nonfinite_effect_is_rejected():
    with pytest.raises(ValueError):
        meta_analyze(yi=[0.1, float("nan"), 0.5], sei=[0.1, 0.1, 0.1],
                     method="DL")


@pytest.mark.parametrize("bad", [-0.2, 0.0, 1.0, 1.5])
def test_invalid_confidence_level_is_rejected(bad):
    # level outside (0, 1) yields inverted / nan / infinite CIs if unchecked.
    with pytest.raises(ValueError):
        meta_analyze(yi=YI, sei=SEI, method="DL", level=bad)


def test_invalid_test_raises():
    with pytest.raises(ValueError):
        meta_analyze(yi=YI, sei=SEI, method="REML", test="bogus")


def test_knha_summary_flags_hk_on_ci():
    yi, vi = _bcg_effects()
    res = meta_analyze(yi=yi, vi=vi, method="REML", test="knha")
    assert "(HK)" in res.summary()
    assert "(HK)" not in meta_analyze(yi=yi, vi=vi, method="REML").summary()
