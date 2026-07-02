"""Effect-size intake: turn incompletely-reported study statistics into
usable ``(mean, SD)`` and ultimately ``(yi, vi)`` — with full provenance.

Primary studies frequently report a median with a range or interquartile
range instead of a mean and standard deviation, or give only a confidence
interval / p-value / test statistic for an effect. This module recovers the
quantities a meta-analysis needs and, critically, records *how* each number
was produced so an extraction can be audited and reproduced.

Estimators
----------
SD from quantiles — Wan et al. (2014):
    xi(n)  = 2 * Phi^-1((n - 0.375) / (n + 0.25))
    eta(n) = 2 * Phi^-1((0.75 n - 0.125) / (n + 0.25))
    S1 {min, median, max}           SD = (b - a) / xi
    S2 {q1, median, q3}             SD = (q3 - q1) / eta
    S3 {min, q1, median, q3, max}   SD = 0.5 [ (b - a)/xi + (q3 - q1)/eta ]

Mean from quantiles — Luo et al. (2018) optimal weighted estimators (default):
    S1  mu = w1 (a+b)/2 + (1-w1) m,                w1 = 4 / (4 + n^0.75)
    S2  mu = w2 (q1+q3)/2 + (1-w2) m,              w2 = 0.7 + 0.39/n
    S3  mu = w3 (a+b)/2 + w4 (q1+q3)/2 + (1-w3-w4) m,
            w3 = 2.2 / (2.2 + n^0.75),  w4 = 0.7 - 0.72 / n^0.55
The three weights in each scenario sum to 1. Formulas verified against the
R package ``metaBLUE`` (``Luo.mean`` / ``Wan.std``) and pinned in the tests.

A simpler Wan (2014) / Hozo (2005) / Bland (2015) mean is available as an
explicitly-requested fallback (``mean_method="wan"``); it is never the default.

SE recovery — Part B — backs a standard error out of a reported CI, a
two-sided p-value, a t-statistic, or an F(1, df). Each records the
distributional assumption it makes.

Every recovered number is returned as a :class:`Derived` carrying its raw
inputs, the estimator used, its assumptions, and a citation key. Converting a
pair of study arms with :func:`to_effects` threads that lineage into the final
``(yi, vi)``, and :func:`extraction_log` renders it as a supplementary table.

References
----------
- Wan, Wang, Liu & Tong (2014). *Estimating the sample mean and standard
  deviation from the sample size, median, range and/or interquartile range.*
  BMC Med Res Methodol 14:135.
- Luo, Wan, Liu & Tong (2018). *Optimally estimating the sample mean from the
  sample size, median, mid-range, and/or mid-quartile range.* Stat Methods Med
  Res 27(6):1785-1805.
- Hozo, Djulbegovic & Hozo (2005); Bland (2015) — simpler mean fallbacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .effects import effect_md, effect_smd

__all__ = [
    "Derived",
    "GroupSummary",
    "DerivedEffect",
    "mean_sd_from_median_range",
    "mean_sd_from_median_iqr",
    "mean_sd_from_five_number",
    "mean_sd_from_summary",
    "se_from_ci",
    "se_from_pvalue",
    "se_from_tstat",
    "se_from_fstat",
    "to_effects",
    "extraction_log",
]


# ── Provenance containers ──────────────────────────────────────────────
@dataclass
class Derived:
    """A single recovered value together with the record of its derivation.

    Attributes
    ----------
    value : the recovered number.
    source_fields : the raw inputs actually used, e.g. ``{"q1": 4, "q3": 9,
        "n": 40}``.
    method : short estimator key, e.g. ``"Wan2014_S2_SD"``,
        ``"Luo2018_S2_mean"``, ``"SE_from_CI95"``.
    assumptions : distributional/other assumptions the estimator relies on.
    reference : short citation key, e.g. ``"Wan2014"``.
    """

    value: float
    source_fields: dict
    method: str
    assumptions: list[str]
    reference: str


@dataclass
class GroupSummary:
    """Recovered ``(mean, SD, n)`` for one study arm, each with provenance."""

    mean: Derived
    sd: Derived
    n: int
    label: str | None = None


@dataclass
class DerivedEffect:
    """An effect size recovered from converted study statistics.

    Carries the pooled-ready ``(yi, vi)`` plus the full lineage: the list of
    :class:`Derived` means/SDs consumed to produce it.
    """

    yi: float
    vi: float
    measure: str
    method: str
    lineage: list[Derived]
    assumptions: list[str]
    reference: str
    study: str | None = None


# ── Wan (2014) SD normalizers ──────────────────────────────────────────
def _xi(n: float) -> float:
    """Range normalizer xi(n) = 2 Phi^-1((n - 0.375)/(n + 0.25))."""
    return 2.0 * float(stats.norm.ppf((n - 0.375) / (n + 0.25)))


def _eta(n: float) -> float:
    """IQR normalizer eta(n) = 2 Phi^-1((0.75 n - 0.125)/(n + 0.25))."""
    return 2.0 * float(stats.norm.ppf((0.75 * n - 0.125) / (n + 0.25)))


def _check_n(n) -> int:
    if not (isinstance(n, (int, np.integer)) or float(n).is_integer()):
        raise ValueError(f"n must be a whole number, got {n!r}")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return n


def _require_positive_normalizer(name: str, value: float, n: int) -> float:
    # ppf argument is in (0,1) for all n >= 1, but hits 0.5 at n = 1 which
    # makes the normalizer exactly zero (SD would be infinite/undefined).
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name}(n) is non-positive at n={n}; the SD estimator needs "
            f"n >= 2 for a finite standard deviation"
        )
    return value


def _ordered(*pairs):
    """Raise if the named quantities are not non-decreasing left to right."""
    prev_name, prev = pairs[0]
    for name, val in pairs[1:]:
        if val < prev:
            raise ValueError(
                f"impossible ordering: {prev_name}={prev} > {name}={val}; "
                f"quantiles/min/median/max must satisfy "
                f"min <= q1 <= median <= q3 <= max"
            )
        prev_name, prev = name, val


# ── Part A: mean/SD from quantiles ─────────────────────────────────────
def _luo_or_fallback_mean_s1(a, m, b, n, mean_method):
    if mean_method == "luo":
        w1 = 4.0 / (4.0 + n ** 0.75)
        value = w1 * (a + b) / 2.0 + (1.0 - w1) * m
        return value, "Luo2018_S1_mean", "Luo2018", ["underlying data ~normal"]
    if mean_method == "wan":
        # Hozo/Wan simple estimator: (a + 2 median + b) / 4.
        value = (a + 2.0 * m + b) / 4.0
        return value, "Wan2014_S1_mean", "Hozo2005", [
            "underlying data ~normal", "simpler unweighted fallback"
        ]
    raise ValueError(f"mean_method must be 'luo' or 'wan', got {mean_method!r}")


def _luo_or_fallback_mean_s2(q1, m, q3, n, mean_method):
    if mean_method == "luo":
        w2 = 0.7 + 0.39 / n
        value = w2 * (q1 + q3) / 2.0 + (1.0 - w2) * m
        return value, "Luo2018_S2_mean", "Luo2018", ["underlying data ~normal"]
    if mean_method == "wan":
        # Wan simple estimator: (q1 + median + q3) / 3.
        value = (q1 + m + q3) / 3.0
        return value, "Wan2014_S2_mean", "Wan2014", [
            "underlying data ~normal", "simpler unweighted fallback"
        ]
    raise ValueError(f"mean_method must be 'luo' or 'wan', got {mean_method!r}")


def _luo_or_fallback_mean_s3(a, q1, m, q3, b, n, mean_method):
    if mean_method == "luo":
        w3 = 2.2 / (2.2 + n ** 0.75)
        w4 = 0.7 - 0.72 / n ** 0.55
        value = (w3 * (a + b) / 2.0
                 + w4 * (q1 + q3) / 2.0
                 + (1.0 - w3 - w4) * m)
        return value, "Luo2018_S3_mean", "Luo2018", ["underlying data ~normal"]
    if mean_method == "wan":
        # Bland simple estimator: (a + 2 q1 + 2 median + 2 q3 + b) / 8.
        value = (a + 2.0 * q1 + 2.0 * m + 2.0 * q3 + b) / 8.0
        return value, "Wan2014_S3_mean", "Bland2015", [
            "underlying data ~normal", "simpler unweighted fallback"
        ]
    raise ValueError(f"mean_method must be 'luo' or 'wan', got {mean_method!r}")


def mean_sd_from_median_range(a, m, b, n, mean_method: str = "luo",
                              label: str | None = None) -> GroupSummary:
    """S1: recover ``(mean, SD)`` from {min ``a``, median ``m``, max ``b``, ``n``}.

    SD uses Wan (2014); mean uses Luo (2018) by default (``mean_method="wan"``
    selects the simpler Hozo/Wan fallback). Assumes the underlying data are
    approximately normal.
    """
    a, m, b = float(a), float(m), float(b)
    n = _check_n(n)
    _ordered(("min", a), ("median", m), ("max", b))
    xi = _require_positive_normalizer("xi", _xi(n), n)

    sd_val = (b - a) / xi
    sd = Derived(sd_val, {"min": a, "max": b, "n": n},
                 "Wan2014_S1_SD", ["underlying data ~normal"], "Wan2014")

    mval, mmeth, mref, massum = _luo_or_fallback_mean_s1(a, m, b, n, mean_method)
    mean = Derived(mval, {"min": a, "median": m, "max": b, "n": n},
                   mmeth, massum, mref)
    return GroupSummary(mean=mean, sd=sd, n=n, label=label)


def mean_sd_from_median_iqr(q1, m, q3, n, mean_method: str = "luo",
                            label: str | None = None) -> GroupSummary:
    """S2: recover ``(mean, SD)`` from {``q1``, median ``m``, ``q3``, ``n``}.

    SD uses Wan (2014); mean uses Luo (2018) by default. Assumes approximate
    normality of the underlying data.
    """
    q1, m, q3 = float(q1), float(m), float(q3)
    n = _check_n(n)
    _ordered(("q1", q1), ("median", m), ("q3", q3))
    eta = _require_positive_normalizer("eta", _eta(n), n)

    sd_val = (q3 - q1) / eta
    sd = Derived(sd_val, {"q1": q1, "q3": q3, "n": n},
                 "Wan2014_S2_SD", ["underlying data ~normal"], "Wan2014")

    mval, mmeth, mref, massum = _luo_or_fallback_mean_s2(q1, m, q3, n, mean_method)
    mean = Derived(mval, {"q1": q1, "median": m, "q3": q3, "n": n},
                   mmeth, massum, mref)
    return GroupSummary(mean=mean, sd=sd, n=n, label=label)


def mean_sd_from_five_number(a, q1, m, q3, b, n, mean_method: str = "luo",
                             label: str | None = None) -> GroupSummary:
    """S3: recover ``(mean, SD)`` from the five-number summary + ``n``.

    Inputs {min ``a``, ``q1``, median ``m``, ``q3``, max ``b``}. SD averages the
    range- and IQR-based Wan (2014) estimators; mean uses Luo (2018) by default.
    Assumes approximate normality of the underlying data.
    """
    a, q1, m, q3, b = float(a), float(q1), float(m), float(q3), float(b)
    n = _check_n(n)
    _ordered(("min", a), ("q1", q1), ("median", m), ("q3", q3), ("max", b))
    xi = _require_positive_normalizer("xi", _xi(n), n)
    eta = _require_positive_normalizer("eta", _eta(n), n)

    sd_val = 0.5 * ((b - a) / xi + (q3 - q1) / eta)
    sd = Derived(sd_val, {"min": a, "q1": q1, "q3": q3, "max": b, "n": n},
                 "Wan2014_S3_SD", ["underlying data ~normal"], "Wan2014")

    mval, mmeth, mref, massum = _luo_or_fallback_mean_s3(
        a, q1, m, q3, b, n, mean_method)
    mean = Derived(mval, {"min": a, "q1": q1, "median": m, "q3": q3,
                          "max": b, "n": n}, mmeth, massum, mref)
    return GroupSummary(mean=mean, sd=sd, n=n, label=label)


def mean_sd_from_summary(n, minimum=None, q1=None, median=None, q3=None,
                         maximum=None, mean_method: str = "luo",
                         label: str | None = None) -> GroupSummary:
    """Dispatch to S1/S2/S3 based on which summary statistics are provided.

    Requires ``median``. {min, max} -> S1; {q1, q3} -> S2; all four plus median
    -> S3. Raises if the combination does not map to a supported scenario.
    """
    if median is None:
        raise ValueError("median is required")
    have_range = minimum is not None and maximum is not None
    have_iqr = q1 is not None and q3 is not None
    if have_range and have_iqr:
        return mean_sd_from_five_number(minimum, q1, median, q3, maximum, n,
                                        mean_method=mean_method, label=label)
    if have_iqr:
        return mean_sd_from_median_iqr(q1, median, q3, n,
                                       mean_method=mean_method, label=label)
    if have_range:
        return mean_sd_from_median_range(minimum, median, maximum, n,
                                         mean_method=mean_method, label=label)
    raise ValueError(
        "insufficient fields: provide {min, max} (S1), {q1, q3} (S2), or all "
        "four with median (S3)"
    )


# ── Part B: SE recovery from summary statistics ────────────────────────
def _z_for_level(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return float(stats.norm.ppf(0.5 + level / 2.0))


def se_from_ci(lower, upper, level: float = 0.95,
               log_scale: bool = False) -> Derived:
    """Standard error from a two-sided confidence interval.

    ``SE = (upper - lower) / (2 * z_level)`` where ``z_level = Phi^-1(0.5 +
    level/2)``. Assumes the estimator is normally distributed on the analysis
    scale and the CI is symmetric about the point estimate.

    For ratio measures (odds/risk/hazard ratios) set ``log_scale=True``: the
    CI is logged first and the SE is returned on the log scale,
    ``SE = (ln(upper) - ln(lower)) / (2 z)``.
    """
    lower, upper = float(lower), float(upper)
    if upper <= lower:
        raise ValueError(f"need upper > lower, got [{lower}, {upper}]")
    z = _z_for_level(level)
    pct = int(round(level * 100))
    if log_scale:
        if lower <= 0:
            raise ValueError("log-scale CI bounds must be positive")
        value = (math.log(upper) - math.log(lower)) / (2.0 * z)
        return Derived(
            value, {"lower": lower, "upper": upper, "level": level},
            f"SE_from_CI{pct}_log",
            ["estimator normal on the log scale", "symmetric log CI"],
            "Altman2011",
        )
    value = (upper - lower) / (2.0 * z)
    return Derived(
        value, {"lower": lower, "upper": upper, "level": level},
        f"SE_from_CI{pct}",
        ["estimator normal on the analysis scale", "symmetric CI"],
        "Altman2011",
    )


def se_from_pvalue(estimate, p, log_scale: bool = False) -> Derived:
    """Standard error from a two-sided p-value and a point estimate.

    ``z = Phi^-1(1 - p/2)`` and ``SE = |estimate| / z``. Assumes a normal
    (Wald) test of the estimate against zero. For ratio measures set
    ``log_scale=True`` to interpret ``estimate`` as a ratio and work with
    ``ln(estimate)`` (SE on the log scale).
    """
    estimate, p = float(estimate), float(p)
    if not 0 < p < 1:
        raise ValueError(f"p must be in the open interval (0, 1), got {p}")
    beta = math.log(estimate) if log_scale else estimate
    if beta == 0:
        raise ValueError("estimate is zero; SE is undefined from a p-value")
    z = float(stats.norm.ppf(1.0 - p / 2.0))
    value = abs(beta) / z
    assumptions = ["two-sided Wald test", "estimator normally distributed"]
    if log_scale:
        assumptions.append("estimate and SE on the log scale")
    return Derived(
        value, {"estimate": estimate, "p": p},
        "SE_from_pvalue_log" if log_scale else "SE_from_pvalue",
        assumptions, "Altman2011",
    )


def se_from_tstat(estimate, t, df) -> Derived:
    """Standard error from a t-statistic testing the estimate against zero.

    Since ``t = estimate / SE`` exactly, ``SE = |estimate / t|``; the degrees
    of freedom fix the reference distribution (and any p-value) but do not
    enter the SE. Assumes ``t`` is the Wald-type statistic for this estimate.
    """
    estimate, t, df = float(estimate), float(t), float(df)
    if t == 0:
        raise ValueError("t-statistic is zero; SE is undefined")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    value = abs(estimate / t)
    return Derived(
        value, {"estimate": estimate, "t": t, "df": df},
        "SE_from_tstat",
        ["t = estimate / SE (Wald statistic)", "SE independent of df"],
        "Altman2011",
    )


def se_from_fstat(estimate, f, df) -> Derived:
    """Standard error from an ``F(1, df)`` statistic and a point estimate.

    An ``F(1, df)`` equals ``t^2`` with ``t ~ t(df)``, so ``|t| = sqrt(F)`` and
    ``SE = |estimate| / sqrt(F)``. The sign of the effect is lost by ``F`` and
    is taken from ``estimate``. Assumes a single-degree-of-freedom contrast.
    """
    estimate, f, df = float(estimate), float(f), float(df)
    if f <= 0:
        raise ValueError(f"F statistic must be positive, got {f}")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    value = abs(estimate) / math.sqrt(f)
    return Derived(
        value, {"estimate": estimate, "F": f, "df": df},
        "SE_from_Fstat",
        ["F(1, df) = t^2", "single-df contrast", "sign taken from estimate"],
        "Altman2011",
    )


# ── Part C: (mean, SD, n) per arm -> (yi, vi) with lineage ─────────────
_CONTINUOUS = {"MD": effect_md, "SMD": effect_smd}


def to_effects(measure: str, group1: GroupSummary, group2: GroupSummary,
               study: str | None = None, **kwargs) -> DerivedEffect:
    """Convert two recovered arm summaries into ``(yi, vi)`` with provenance.

    ``measure`` is a continuous two-group effect: ``"MD"`` (raw mean difference)
    or ``"SMD"`` (standardized, Hedges' g). Extra keyword arguments are passed
    through to the underlying ``effects.py`` function (e.g. ``correct=False``
    for SMD). The returned :class:`DerivedEffect` threads the four consumed
    :class:`Derived` means/SDs as its ``lineage``.
    """
    key = str(measure).upper()
    if key not in _CONTINUOUS:
        raise ValueError(
            f"to_effects supports continuous measures {sorted(_CONTINUOUS)}, "
            f"got {measure!r}; use effects.py directly for binary outcomes"
        )
    fn = _CONTINUOUS[key]
    yi, vi = fn(m1=group1.mean.value, sd1=group1.sd.value, n1=group1.n,
                m2=group2.mean.value, sd2=group2.sd.value, n2=group2.n,
                **kwargs)
    yi, vi = float(np.asarray(yi)), float(np.asarray(vi))

    lineage = [group1.mean, group1.sd, group2.mean, group2.sd]
    assumptions = sorted({a for d in lineage for a in d.assumptions})
    references = sorted({d.reference for d in lineage})
    return DerivedEffect(
        yi=yi, vi=vi, measure=key,
        method=f"{key}_from_recovered_mean_sd",
        lineage=lineage,
        assumptions=assumptions,
        reference="+".join(references),
        study=study,
    )


def extraction_log(*items, as_dataframe: bool = False):
    """Render a per-quantity extraction log for a supplementary table.

    Accepts any mix of :class:`DerivedEffect`, :class:`GroupSummary`, and bare
    :class:`Derived` objects and flattens them into one row per recovered
    quantity, with columns: study, arm, quantity, value, method, reference,
    assumptions, source_fields. Returns a list of dicts, or a pandas DataFrame
    if ``as_dataframe=True``.
    """
    rows: list[dict] = []

    def add(d: Derived, *, study=None, arm=None, quantity=None):
        rows.append({
            "study": study,
            "arm": arm,
            "quantity": quantity,
            "value": d.value,
            "method": d.method,
            "reference": d.reference,
            "assumptions": "; ".join(d.assumptions),
            "source_fields": dict(d.source_fields),
        })

    def add_group(g: GroupSummary, *, study=None, arm=None):
        add(g.mean, study=study, arm=arm or g.label, quantity="mean")
        add(g.sd, study=study, arm=arm or g.label, quantity="SD")

    for item in items:
        if isinstance(item, DerivedEffect):
            # lineage is [g1.mean, g1.sd, g2.mean, g2.sd]
            names = ["group1", "group1", "group2", "group2"]
            quants = ["mean", "SD", "mean", "SD"]
            for d, arm, q in zip(item.lineage, names, quants):
                add(d, study=item.study, arm=arm, quantity=q)
            rows.append({
                "study": item.study,
                "arm": None,
                "quantity": f"{item.measure} (yi, vi)",
                "value": item.yi,
                "method": item.method,
                "reference": item.reference,
                "assumptions": "; ".join(item.assumptions),
                "source_fields": {"yi": item.yi, "vi": item.vi},
            })
        elif isinstance(item, GroupSummary):
            add_group(item, study=item.label, arm=item.label)
        elif isinstance(item, Derived):
            add(item, quantity=item.method)
        else:
            raise TypeError(f"cannot log object of type {type(item).__name__}")

    if as_dataframe:
        import pandas as pd
        return pd.DataFrame(rows)
    return rows
