"""Compute effect sizes and their sampling variances from raw study data.

Each function takes array-like columns and returns ``(yi, vi)`` as numpy
arrays: the study effect sizes and their sampling variances, ready to pass
to :func:`metanalysis.meta_analyze`.

Continuous outcomes (two independent groups):
    effect_md   raw mean difference
    effect_smd  standardized mean difference (Hedges' g, with correction)

Binary outcomes (events / group totals):
    effect_or   log odds ratio
    effect_rr   log risk ratio
    effect_rd   risk difference

Correlations:
    effect_cor  Fisher's z transform
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effect_md", "effect_smd", "effect_or", "effect_rr", "effect_rd",
    "effect_cor", "compute_effects",
]


def _arr(x):
    return np.asarray(x, dtype=float)


def effect_md(m1, sd1, n1, m2, sd2, n2):
    """Raw mean difference: yi = m1 - m2, vi = sd1^2/n1 + sd2^2/n2."""
    m1, sd1, n1, m2, sd2, n2 = map(_arr, (m1, sd1, n1, m2, sd2, n2))
    yi = m1 - m2
    vi = sd1 ** 2 / n1 + sd2 ** 2 / n2
    return yi, vi


def effect_smd(m1, sd1, n1, m2, sd2, n2, correct=True):
    """Standardized mean difference.

    Cohen's d uses the pooled within-group SD. With ``correct=True`` (default)
    Hedges' small-sample correction J is applied, yielding Hedges' g. Variance
    follows Borenstein et al. (2009), eq. 4.24: ``V_g = J^2 * V_d`` (the ``J^2``
    multiplies the whole ``V_d``). metafor's default does not put ``J^2`` on the
    ``1/n1 + 1/n2`` term, so its SMD variances run ~1-2% larger; both are
    standard. ``J = 1 - 3/(4*df - 1)`` is Hedges' approximation to the exact
    gamma-ratio correction (agree to ~1e-6).
    """
    m1, sd1, n1, m2, sd2, n2 = map(_arr, (m1, sd1, n1, m2, sd2, n2))
    df = n1 + n2 - 2
    if np.any(df < 1):
        raise ValueError(
            "SMD requires n1 + n2 > 2 (df = n1 + n2 - 2 >= 1) so the pooled "
            "within-group SD is defined"
        )
    sp = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df)
    d = (m1 - m2) / sp
    vd = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    if correct:
        J = 1 - 3 / (4 * df - 1)
        return J * d, J ** 2 * vd
    return d, vd


def _cells(e1, n1, e2, n2, add):
    """2x2 cells (a,b,c,d) with continuity correction on zero-cell studies."""
    e1, n1, e2, n2 = map(_arr, (e1, n1, e2, n2))
    a = e1
    b = n1 - e1
    c = e2
    d = n2 - e2
    if add:
        zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        a = a + add * zero
        b = b + add * zero
        c = c + add * zero
        d = d + add * zero
    return a, b, c, d


def effect_or(e1, n1, e2, n2, add=0.5):
    """Log odds ratio.

    yi = ln(ad / bc), vi = 1/a + 1/b + 1/c + 1/d, where a/b are events/
    non-events in group 1 and c/d in group 2. Studies with any zero cell get
    ``add`` (default 0.5) added to all four cells.

    The 0.5 / only-zero-studies continuity correction reproduces metafor's
    default (``add=1/2, to="only0"``). It is the accepted default but is known
    to be biased for sparse or unbalanced data (Sweeting et al. 2004); for many
    zero cells consider Peto OR or exact methods. ``add=0`` opts out entirely
    and then requires zero-cell-free data (a zero cell yields +-inf).
    """
    a, b, c, d = _cells(e1, n1, e2, n2, add)
    yi = np.log(a * d / (b * c))
    vi = 1 / a + 1 / b + 1 / c + 1 / d
    return yi, vi


def effect_rr(e1, n1, e2, n2, add=0.5):
    """Log risk (rate) ratio.

    yi = ln((a/n1) / (c/n2)), vi = 1/a - 1/n1 + 1/c - 1/n2. Studies with any
    zero cell get ``add`` added to all four cells.
    """
    a, b, c, d = _cells(e1, n1, e2, n2, add)
    n1c = a + b
    n2c = c + d
    yi = np.log((a / n1c) / (c / n2c))
    vi = 1 / a - 1 / n1c + 1 / c - 1 / n2c
    return yi, vi


def effect_rd(e1, n1, e2, n2):
    """Risk difference.

    yi = a/n1 - c/n2, vi = a*b/n1^3 + c*d/n2^3. No continuity correction is
    needed (the estimator is well defined for zero cells).
    """
    e1, n1, e2, n2 = map(_arr, (e1, n1, e2, n2))
    a, c = e1, e2
    b, d = n1 - e1, n2 - e2
    yi = a / n1 - c / n2
    vi = a * b / n1 ** 3 + c * d / n2 ** 3
    return yi, vi


def effect_cor(r, n):
    """Fisher's z transform of a correlation.

    yi = arctanh(r), vi = 1/(n - 3). Pool on the z scale, then back-transform
    the pooled estimate with tanh for reporting.
    """
    r, n = _arr(r), _arr(n)
    if np.any(np.abs(r) >= 1):
        raise ValueError("correlations must satisfy -1 < r < 1 (arctanh is "
                         "infinite at +-1)")
    if np.any(n <= 3):
        raise ValueError("Fisher-z variance 1/(n-3) requires n > 3")
    yi = np.arctanh(r)
    vi = 1.0 / (n - 3)
    return yi, vi


_MEASURES = {
    "MD": effect_md,
    "SMD": effect_smd,
    "OR": effect_or,
    "RR": effect_rr,
    "RD": effect_rd,
    "COR": effect_cor,
}


def compute_effects(measure, **cols):
    """Compute ``(yi, vi)`` for a named effect-size ``measure``.

    Dispatches to the matching ``effect_*`` function, forwarding column
    keyword arguments (which may be lists, numpy arrays, or pandas Series).

    measure : {"MD", "SMD", "OR", "RR", "RD", "COR"} (case-insensitive)
        MD  raw mean difference           -> m1, sd1, n1, m2, sd2, n2
        SMD standardized mean difference  -> m1, sd1, n1, m2, sd2, n2 [, correct]
        OR  log odds ratio                -> e1, n1, e2, n2 [, add]
        RR  log risk ratio                -> e1, n1, e2, n2 [, add]
        RD  risk difference               -> e1, n1, e2, n2
        COR Fisher's z correlation        -> r, n
    """
    key = str(measure).upper()
    if key not in _MEASURES:
        raise ValueError(
            f"unknown measure {measure!r}; choose from {sorted(_MEASURES)}"
        )
    return _MEASURES[key](**cols)
