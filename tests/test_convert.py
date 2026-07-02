"""Tests for the effect-size intake / conversion layer (convert.py).

Reference values for the Wan (2014) SD and Luo (2018) mean estimators are
generated from the R package ``metaBLUE`` 0.1.0 (``Luo.mean`` / ``Wan.std``),
which implements the published formulas verbatim. They are hard-coded here and
asserted to well beyond 3 significant figures.

metaBLUE reference calls (X ordered as the scenario lists its fields):
    Luo.mean(c(2,5,12), 25, "S1")$muhat      = 5.5269974229
    Wan.std (c(2,5,12), 25, "S1")$sigmahat   = 2.5455437980
    Luo.mean(c(4,6,9),  40, "S2")$muhat      = 6.3548750000
    Wan.std (c(4,6,9),  40, "S2")$sigmahat   = 3.8446495074
    Luo.mean(c(1,4,6,9,15), 50, "S3")$muhat  = 6.5176270017
    Wan.std (c(1,4,6,9,15), 50, "S3")$sigmahat = 3.4684429367
    2*qnorm((25-0.375)/(25+0.25))            = 3.9284336839   (xi at n=25)
    2*qnorm((0.75*40-0.125)/(40+0.25))       = 1.3005086655   (eta at n=40)
"""

import math

import numpy as np
import pytest

from metanalysis import meta_analyze
from metanalysis.convert import (
    Derived,
    DerivedEffect,
    GroupSummary,
    extraction_log,
    mean_sd_from_five_number,
    mean_sd_from_median_iqr,
    mean_sd_from_median_range,
    mean_sd_from_summary,
    se_from_ci,
    se_from_fstat,
    se_from_pvalue,
    se_from_tstat,
    to_effects,
)


# ── Part A: Wan SD against metaBLUE ────────────────────────────────────
def test_wan_sd_s1_matches_metablue():
    g = mean_sd_from_median_range(a=2, m=5, b=12, n=25)
    assert g.sd.value == pytest.approx(2.5455437980, rel=1e-9)


def test_wan_sd_s2_matches_metablue():
    g = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    assert g.sd.value == pytest.approx(3.8446495074, rel=1e-9)


def test_wan_sd_s3_matches_metablue():
    g = mean_sd_from_five_number(a=1, q1=4, m=6, q3=9, b=15, n=50)
    assert g.sd.value == pytest.approx(3.4684429367, rel=1e-9)


def test_wan_normalizers_match_definition():
    # xi(25) and eta(40) reproduce the reported Wan constants.
    from scipy import stats
    xi = 2 * stats.norm.ppf((25 - 0.375) / (25 + 0.25))
    eta = 2 * stats.norm.ppf((0.75 * 40 - 0.125) / (40 + 0.25))
    assert xi == pytest.approx(3.9284336839, rel=1e-9)
    assert eta == pytest.approx(1.3005086655, rel=1e-9)


# ── Part A: Luo mean against metaBLUE ──────────────────────────────────
def test_luo_mean_s1_matches_metablue():
    g = mean_sd_from_median_range(a=2, m=5, b=12, n=25)
    assert g.mean.value == pytest.approx(5.5269974229, rel=1e-9)
    assert "Luo2018" in g.mean.method


def test_luo_mean_s2_matches_metablue():
    g = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    assert g.mean.value == pytest.approx(6.3548750000, rel=1e-9)


def test_luo_mean_s3_matches_metablue():
    g = mean_sd_from_five_number(a=1, q1=4, m=6, q3=9, b=15, n=50)
    assert g.mean.value == pytest.approx(6.5176270017, rel=1e-9)


def test_luo_weights_sum_to_one_s3():
    # The three Luo S3 weights must sum to 1: with all quantiles equal the
    # estimated mean equals that common value.
    g = mean_sd_from_five_number(a=7, q1=7, m=7, q3=7, b=7, n=33)
    assert g.mean.value == pytest.approx(7.0, abs=1e-12)


def test_wan_mean_fallback_is_labeled_and_not_default():
    # Fallback Wan/Bland means differ from Luo and must be explicitly requested.
    default = mean_sd_from_median_range(a=2, m=5, b=12, n=25)
    fallback = mean_sd_from_median_range(a=2, m=5, b=12, n=25, mean_method="wan")
    assert "Luo2018" in default.mean.method
    assert "Wan2014" in fallback.mean.method or "Hozo" in fallback.mean.method
    # Wan S1 mean = (a + 2m + b)/4 = (2 + 10 + 12)/4 = 6.0
    assert fallback.mean.value == pytest.approx(6.0, rel=1e-12)
    assert fallback.mean.value != default.mean.value


def test_dispatcher_selects_scenario_by_available_fields():
    s1 = mean_sd_from_summary(n=25, minimum=2, median=5, maximum=12)
    s2 = mean_sd_from_summary(n=40, q1=4, median=6, q3=9)
    s3 = mean_sd_from_summary(n=50, minimum=1, q1=4, median=6, q3=9, maximum=15)
    assert s1.sd.value == pytest.approx(2.5455437980, rel=1e-9)
    assert s2.mean.value == pytest.approx(6.3548750000, rel=1e-9)
    assert s3.mean.value == pytest.approx(6.5176270017, rel=1e-9)


# ── Part A: guards ─────────────────────────────────────────────────────
def test_impossible_ordering_raises():
    with pytest.raises(ValueError):
        mean_sd_from_median_iqr(q1=9, m=6, q3=4, n=40)      # q1 > q3
    with pytest.raises(ValueError):
        mean_sd_from_median_range(a=2, m=20, b=12, n=25)    # m > b
    with pytest.raises(ValueError):
        mean_sd_from_five_number(a=1, q1=4, m=6, q3=9, b=5, n=50)  # b < q3


def test_n_too_small_for_normalizer_raises():
    # n = 1 makes the SD normalizer zero (ppf argument = 0.5).
    with pytest.raises(ValueError):
        mean_sd_from_median_range(a=2, m=5, b=12, n=1)


# ── Part A: Monte-Carlo round trip ─────────────────────────────────────
def test_round_trip_recovers_mean_and_sd_from_normal_data():
    rng = np.random.default_rng(20240607)
    mu, sigma, n, reps = 50.0, 8.0, 200, 400
    m_hat, s_hat = [], []
    for _ in range(reps):
        x = np.sort(rng.normal(mu, sigma, n))
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        g = mean_sd_from_five_number(a=x[0], q1=q1, m=med, q3=q3, b=x[-1], n=n)
        m_hat.append(g.mean.value)
        s_hat.append(g.sd.value)
    # Averaged over replications the estimators are ~unbiased for normal data.
    assert np.mean(m_hat) == pytest.approx(mu, abs=0.15)
    assert np.mean(s_hat) == pytest.approx(sigma, rel=0.03)


# ── Part B: SE recovery ────────────────────────────────────────────────
def test_se_from_ci_round_trip():
    est, se = 0.40, 0.125
    z = 1.959963985
    d = se_from_ci(lower=est - z * se, upper=est + z * se, level=0.95)
    assert d.value == pytest.approx(se, rel=1e-9)


def test_se_from_ci_arbitrary_level():
    est, se, level = -1.2, 0.3, 0.90
    from scipy import stats
    z = stats.norm.ppf(0.5 + level / 2)
    d = se_from_ci(lower=est - z * se, upper=est + z * se, level=level)
    assert d.value == pytest.approx(se, rel=1e-9)


def test_se_from_ci_log_scale_for_ratio_measures():
    # A ratio CI must be logged first; SE is on the log scale.
    from scipy import stats
    lo, hi = 1.5, 3.0
    z = float(stats.norm.ppf(0.975))
    d = se_from_ci(lower=lo, upper=hi, level=0.95, log_scale=True)
    assert d.value == pytest.approx((math.log(hi) - math.log(lo)) / (2 * z),
                                    rel=1e-12)
    assert any("log" in a.lower() for a in d.assumptions)


def test_se_from_pvalue_round_trip():
    est, se = 0.30, 0.10
    from scipy import stats
    z_true = est / se
    p = 2 * stats.norm.sf(abs(z_true))
    d = se_from_pvalue(estimate=est, p=p)
    assert d.value == pytest.approx(se, rel=1e-9)


def test_se_from_pvalue_guards_degenerate_p():
    with pytest.raises(ValueError):
        se_from_pvalue(estimate=0.3, p=0.0)
    with pytest.raises(ValueError):
        se_from_pvalue(estimate=0.3, p=1.0)


def test_se_from_tstat_round_trip():
    est, se = 2.5, 0.8
    t = est / se
    d = se_from_tstat(estimate=est, t=t, df=42)
    assert d.value == pytest.approx(se, rel=1e-12)


def test_se_from_fstat_round_trip():
    # F(1, df) = t^2, so SE = |estimate| / sqrt(F).
    est, se = 2.5, 0.8
    f = (est / se) ** 2
    d = se_from_fstat(estimate=est, f=f, df=42)
    assert d.value == pytest.approx(se, rel=1e-12)


# ── Part C: provenance ─────────────────────────────────────────────────
def _has_full_provenance(d: Derived):
    return (isinstance(d, Derived)
            and d.method and d.reference
            and isinstance(d.assumptions, list) and len(d.assumptions) > 0
            and isinstance(d.source_fields, dict) and len(d.source_fields) > 0)


def test_every_derived_carries_full_provenance():
    items = [
        mean_sd_from_median_range(a=2, m=5, b=12, n=25).mean,
        mean_sd_from_median_range(a=2, m=5, b=12, n=25).sd,
        mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40).mean,
        mean_sd_from_five_number(a=1, q1=4, m=6, q3=9, b=15, n=50).sd,
        se_from_ci(lower=0.1, upper=0.7, level=0.95),
        se_from_pvalue(estimate=0.3, p=0.01),
        se_from_tstat(estimate=2.5, t=3.1, df=40),
        se_from_fstat(estimate=2.5, f=9.6, df=40),
    ]
    for d in items:
        assert _has_full_provenance(d), d


def test_source_fields_name_the_actual_inputs():
    g = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    assert set(g.sd.source_fields) >= {"q1", "q3", "n"}
    assert g.sd.source_fields["q1"] == 4 and g.sd.source_fields["q3"] == 9


# ── Part C: to_effects lineage ─────────────────────────────────────────
def test_to_effects_matches_effect_md_and_threads_lineage():
    from metanalysis.effects import effect_md
    g1 = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    g2 = mean_sd_from_median_iqr(q1=2, m=4, q3=7, n=38)
    eff = to_effects("MD", g1, g2, study="Trial A")

    yi, vi = effect_md(m1=g1.mean.value, sd1=g1.sd.value, n1=40,
                       m2=g2.mean.value, sd2=g2.sd.value, n2=38)
    assert eff.yi == pytest.approx(float(yi), rel=1e-12)
    assert eff.vi == pytest.approx(float(vi), rel=1e-12)

    # Lineage must reference every raw field consumed by both arms.
    consumed = set()
    for d in eff.lineage:
        consumed |= set(d.source_fields)
    assert {"q1", "median", "q3", "n"} <= consumed
    # The recovered (yi, vi) must flow into the validated pooler unchanged.
    res = meta_analyze(yi=[eff.yi], vi=[eff.vi], method="FE")
    assert res.estimate == pytest.approx(eff.yi, rel=1e-12)


def test_to_effects_supports_smd():
    g1 = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    g2 = mean_sd_from_median_iqr(q1=2, m=4, q3=7, n=38)
    eff = to_effects("SMD", g1, g2)
    assert isinstance(eff, DerivedEffect)
    assert np.isfinite(eff.yi) and eff.vi > 0


def test_extraction_log_renders_supplementary_rows():
    g1 = mean_sd_from_median_iqr(q1=4, m=6, q3=9, n=40)
    g2 = mean_sd_from_median_iqr(q1=2, m=4, q3=7, n=38)
    eff = to_effects("MD", g1, g2, study="Trial A")
    rows = extraction_log(eff)
    assert isinstance(rows, list) and len(rows) >= 4      # 2 means + 2 SDs
    needed = {"study", "quantity", "value", "method", "reference", "assumptions"}
    for r in rows:
        assert needed <= set(r)
        assert r["method"] and r["reference"]
