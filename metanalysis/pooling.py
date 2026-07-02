"""Core meta-analytic pooling: fixed-effect and random-effects models.

Implements inverse-variance fixed-effect pooling and random-effects pooling
via DerSimonian-Laird (moment estimator) and REML (restricted maximum
likelihood, iterative). Reports the pooled estimate with a confidence
interval, Cochran's Q, I-squared, H-squared, tau-squared, and a
prediction interval for random-effects models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

VALID_METHODS = ("FE", "DL", "REML")


@dataclass
class MetaResult:
    """Result of a meta-analysis.

    Attributes
    ----------
    method : {"FE", "DL", "REML"}
    k : number of studies
    estimate, se : pooled effect and its standard error
    ci_low, ci_high, level : confidence interval and its coverage
    z, pval : Wald test of the pooled effect against zero
    Q, Q_df, Q_pval : Cochran's heterogeneity test
    I2 : proportion of total variation due to heterogeneity (0-1)
    H2 : Q / df
    tau2, tau : between-study variance used for pooling, and its square root
    pi_low, pi_high : prediction interval (None for fixed effect or k < 3)
    yi, vi, weights : per-study effects, variances, and pooling weights (%)
    """

    method: str
    k: int
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    level: float
    z: float
    pval: float
    Q: float
    Q_df: int
    Q_pval: float
    I2: float
    H2: float
    tau2: float
    tau: float
    pi_low: float | None
    pi_high: float | None
    yi: np.ndarray = field(repr=False)
    vi: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)

    def summary(self) -> str:
        name = {"FE": "Fixed-effect", "DL": "Random-effects (DL)",
                "REML": "Random-effects (REML)"}[self.method]
        pct = int(round(self.level * 100))
        lines = [
            f"{name} meta-analysis  (k = {self.k} studies)",
            f"  Pooled estimate : {self.estimate:.4f}  "
            f"(SE {self.se:.4f})",
            f"  {pct}% CI          : [{self.ci_low:.4f}, {self.ci_high:.4f}]",
            f"  Test of effect  : z = {self.z:.3f},  p = {self.pval:.3g}",
            "  Heterogeneity   : "
            f"Q({self.Q_df}) = {self.Q:.3f}, p = {self.Q_pval:.3g}; "
            f"I2 = {100 * self.I2:.1f}%; tau2 = {self.tau2:.4f}",
        ]
        if self.pi_low is not None:
            lines.append(
                f"  {pct}% pred. int.  : "
                f"[{self.pi_low:.4f}, {self.pi_high:.4f}]"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


def _reml_tau2(yi: np.ndarray, vi: np.ndarray, tau2_init: float,
               max_iter: int = 200, tol: float = 1e-10) -> float:
    """REML between-study variance via fixed-point iteration.

    Uses the estimating equation (Viechtbauer 2005)

        tau2 = sum(w^2 * ((yi - mu)^2 - vi)) / sum(w^2)  +  1 / sum(w)

    with w = 1 / (vi + tau2), iterated to convergence and truncated at 0.
    """
    tau2 = max(0.0, tau2_init)
    for _ in range(max_iter):
        w = 1.0 / (vi + tau2)
        sw = w.sum()
        mu = (w * yi).sum() / sw
        w2 = w * w
        new = (w2 * ((yi - mu) ** 2 - vi)).sum() / w2.sum() + 1.0 / sw
        new = max(0.0, new)
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    return tau2


def meta_analyze(yi, sei=None, vi=None, method: str = "DL",
                 level: float = 0.95) -> MetaResult:
    """Pool effect sizes across studies.

    Parameters
    ----------
    yi : array-like
        Study effect sizes (e.g. log odds ratios, mean differences, Hedges' g).
    sei : array-like, optional
        Standard errors of ``yi``. Provide this or ``vi``.
    vi : array-like, optional
        Sampling variances of ``yi`` (``vi = sei**2``). Provide this or ``sei``.
    method : {"FE", "DL", "REML"}
        "FE" fixed-effect inverse variance; "DL" random-effects
        DerSimonian-Laird; "REML" random-effects restricted maximum likelihood.
    level : float
        Confidence level for intervals (default 0.95).

    Returns
    -------
    MetaResult
    """
    method = method.upper()
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    yi = np.asarray(yi, dtype=float)
    if sei is None and vi is None:
        raise ValueError("provide either sei or vi")
    if vi is None:
        sei = np.asarray(sei, dtype=float)
        vi = sei ** 2
    else:
        vi = np.asarray(vi, dtype=float)
    if yi.shape != vi.shape:
        raise ValueError("yi and sei/vi must have the same length")
    if yi.ndim != 1 or yi.size == 0:
        raise ValueError("yi must be a non-empty 1-D array")
    if np.any(vi <= 0):
        raise ValueError("all variances must be positive")

    k = yi.size
    df = k - 1

    # Fixed-effect (inverse-variance) weights drive the heterogeneity stats.
    wf = 1.0 / vi
    swf = wf.sum()
    mu_f = (wf * yi).sum() / swf
    Q = float((wf * (yi - mu_f) ** 2).sum())

    if df > 0:
        Q_pval = float(stats.chi2.sf(Q, df))
        I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
        H2 = Q / df
    else:
        Q_pval = float("nan")
        I2 = 0.0
        H2 = float("nan")

    # Between-study variance for the chosen model. With a single study
    # (df == 0) heterogeneity is undefined, so there is nothing to estimate.
    if method == "FE" or df < 1:
        tau2 = 0.0
    elif method == "DL":
        C = swf - (wf ** 2).sum() / swf
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    else:  # REML
        C = swf - (wf ** 2).sum() / swf
        dl = max(0.0, (Q - df) / C) if C > 0 else 0.0
        tau2 = _reml_tau2(yi, vi, tau2_init=dl)

    # Pool with total (sampling + between-study) variance.
    w = 1.0 / (vi + tau2)
    sw = w.sum()
    estimate = float((w * yi).sum() / sw)
    se = float(np.sqrt(1.0 / sw))

    zc = float(stats.norm.ppf(0.5 + level / 2))
    ci_low = estimate - zc * se
    ci_high = estimate + zc * se

    z = estimate / se
    pval = float(2 * stats.norm.sf(abs(z)))

    # Prediction interval: only meaningful for random-effects with k >= 3.
    if method != "FE" and k >= 3:
        tc = float(stats.t.ppf(0.5 + level / 2, df=k - 2))
        pi_half = tc * np.sqrt(tau2 + se ** 2)
        pi_low = estimate - pi_half
        pi_high = estimate + pi_half
    else:
        pi_low = pi_high = None

    weights = 100.0 * w / sw

    return MetaResult(
        method=method, k=k, estimate=estimate, se=se,
        ci_low=ci_low, ci_high=ci_high, level=level,
        z=z, pval=pval, Q=Q, Q_df=df, Q_pval=Q_pval,
        I2=I2, H2=H2, tau2=tau2, tau=float(np.sqrt(tau2)),
        pi_low=pi_low, pi_high=pi_high,
        yi=yi, vi=vi, weights=weights,
    )
