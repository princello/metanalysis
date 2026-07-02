"""Cross-check metanalysis against statsmodels on the BCG vaccine dataset.

The Colditz et al. (1994) BCG trials are the canonical meta-analysis example
(metafor's dat.bcg). We compare log-risk-ratio effect sizes and DerSimonian-
Laird pooling against statsmodels' independent implementation. No zero cells,
so agreement should be to ~machine precision on effects and DL statistics.

Run: python validation/cross_check_statsmodels.py
"""

import numpy as np
from statsmodels.stats.meta_analysis import (
    combine_effects,
    effectsize_2proportions,
)

from metanalysis import effect_rr, meta_analyze

# trial: tpos, tneg (vaccinated TB+/TB-), cpos, cneg (control TB+/TB-)
BCG = np.array([
    [4, 119, 11, 128],
    [6, 300, 29, 274],
    [3, 228, 11, 209],
    [62, 13536, 248, 12619],
    [33, 5036, 47, 5761],
    [180, 1361, 372, 1079],
    [8, 2537, 10, 619],
    [505, 87886, 499, 87892],
    [29, 7470, 45, 7232],
    [17, 1699, 65, 1600],
    [186, 50448, 141, 27197],
    [5, 2493, 3, 2338],
    [27, 16886, 29, 17825],
], dtype=float)

tpos, tneg, cpos, cneg = BCG[:, 0], BCG[:, 1], BCG[:, 2], BCG[:, 3]
e1, n1 = tpos, tpos + tneg          # vaccinated
e2, n2 = cpos, cpos + cneg          # control


def report(name, ours, theirs, tol=1e-8):
    diff = abs(ours - theirs)
    ok = "OK " if diff <= tol else "XX "
    print(f"  {ok}{name:28s} ours={ours: .6f}  statsmodels={theirs: .6f}  "
          f"|Δ|={diff:.2e}")
    return diff <= tol


print("BCG vaccine meta-analysis: metanalysis vs statsmodels (log RR)\n")

# --- Effect sizes -----------------------------------------------------------
yi, vi = effect_rr(e1=e1, n1=n1, e2=e2, n2=n2)
sm_eff, sm_var = effectsize_2proportions(tpos, tpos + tneg, cpos, cpos + cneg,
                                         statistic="rr")
print("Effect sizes (log risk ratio):")
all_ok = True
all_ok &= report("max |Δ effect|", 0.0, float(np.max(np.abs(yi - sm_eff))))
all_ok &= report("max |Δ variance|", 0.0, float(np.max(np.abs(vi - sm_var))))

# --- Pooling ----------------------------------------------------------------
res_fe = meta_analyze(yi=yi, vi=vi, method="FE")
res_dl = meta_analyze(yi=yi, vi=vi, method="DL")

cr = combine_effects(sm_eff, sm_var, method_re="chi2")  # chi2 == DL moment
d = cr.summary_frame()  # rows include fixed effect and random effect (DL)

sm_fe = d.loc["fixed effect", "eff"]
sm_re = d.loc["random effect", "eff"]
sm_fe_sd = d.loc["fixed effect", "sd_eff"]
sm_re_sd = d.loc["random effect", "sd_eff"]

print("\nHeterogeneity:")
all_ok &= report("Cochran Q", res_dl.Q, float(cr.q))
all_ok &= report("I^2", res_dl.I2, float(cr.i2))
all_ok &= report("tau^2 (DL)", res_dl.tau2, float(cr.tau2), tol=1e-6)

print("\nFixed-effect pooling:")
all_ok &= report("estimate", res_fe.estimate, float(sm_fe))
all_ok &= report("standard error", res_fe.se, float(sm_fe_sd))

print("\nRandom-effects (DL) pooling:")
all_ok &= report("estimate", res_dl.estimate, float(sm_re), tol=1e-6)
all_ok &= report("standard error", res_dl.se, float(sm_re_sd), tol=1e-6)

print("\nMetanalysis summary (DL):")
print("   ", res_dl.summary().replace("\n", "\n    "))

print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
