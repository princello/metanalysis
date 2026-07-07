"""Small-study effects: Egger's regression test for funnel-plot asymmetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["EggerResult", "egger_test"]


@dataclass
class EggerResult:
    """Result of Egger's regression test.

    ``intercept`` (the bias coefficient) is the asymmetry measure: a value
    far from zero indicates that smaller studies report systematically
    different effects than larger ones. Tested against zero with a t
    distribution on ``df = k - 2`` degrees of freedom.
    """

    intercept: float
    se: float
    t: float
    df: int
    pval: float
    slope: float

    def summary(self) -> str:
        return (
            "Egger's test for funnel-plot asymmetry\n"
            f"  bias (intercept) = {self.intercept:.4f} (SE {self.se:.4f})\n"
            f"  t({self.df}) = {self.t:.3f},  p = {self.pval:.4g}"
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def egger_test(yi, sei=None, vi=None) -> EggerResult:
    """Egger's linear regression test for small-study effects.

    Regresses the standard normal deviate ``yi / sei`` on precision
    ``1 / sei`` by ordinary least squares. The intercept measures funnel
    asymmetry; its t-test (df = k - 2) is Egger's test. Requires k >= 3.

    This is the originally-published Egger et al. (1997) linear-regression test,
    equivalent to metafor's ``regtest(model="lm")``. metafor's *default*
    ``regtest`` is ``model="rma"``, which accounts for residual heterogeneity
    and gives a different (often less significant) result -- so this test will
    match metafor only when metafor is called with ``model="lm"``.

    Parameters
    ----------
    yi : array-like
        Study effect sizes.
    sei, vi : array-like, optional
        Standard errors or variances of ``yi`` (provide exactly one).
    """
    yi = np.asarray(yi, dtype=float)
    if sei is None and vi is None:
        raise ValueError("provide either sei or vi")
    if sei is None:
        sei = np.sqrt(np.asarray(vi, dtype=float))
    else:
        sei = np.asarray(sei, dtype=float)

    k = yi.size
    if k < 3:
        raise ValueError("Egger's test requires at least 3 studies")

    snd = yi / sei           # standard normal deviate (response)
    prec = 1.0 / sei         # precision (predictor)

    # Ordinary least squares of snd on prec with an intercept.
    x_mean = prec.mean()
    y_mean = snd.mean()
    sxx = np.sum((prec - x_mean) ** 2)
    if sxx == 0:
        raise ValueError(
            "Egger's test is undefined when all standard errors are equal: "
            "there is no precision gradient to regress against"
        )
    sxy = np.sum((prec - x_mean) * (snd - y_mean))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    df = k - 2
    resid = snd - (intercept + slope * prec)
    mse = np.sum(resid ** 2) / df
    se_intercept = np.sqrt(mse * (1.0 / k + x_mean ** 2 / sxx))

    t = intercept / se_intercept
    pval = float(2 * stats.t.sf(abs(t), df))

    return EggerResult(
        intercept=float(intercept), se=float(se_intercept), t=float(t),
        df=int(df), pval=pval, slope=float(slope),
    )
