# metanalysis

A small, general-purpose meta-analysis toolkit in Python. Pool study-level
effect sizes with fixed-effect or random-effects models, compute effect sizes
from raw study data, test for small-study effects, and draw forest and funnel
plots.

## Features

- **Pooling**
  - Fixed-effect inverse-variance (`FE`)
  - Random-effects DerSimonian–Laird moment estimator (`DL`)
  - Random-effects REML (`REML`, iterative)
- **Reported for every analysis**: pooled estimate + confidence interval,
  Wald test of the effect, Cochran's `Q` (+ p), `I²`, `H²`, `τ²`/`τ`, and a
  **prediction interval** (random-effects).
- **Effect sizes from raw data**: raw mean difference, standardized mean
  difference (Hedges' *g*), log odds ratio, log risk ratio, risk difference,
  and Fisher's *z* correlation — with continuity correction for zero cells.
- **Publication bias**: Egger's regression test for funnel-plot asymmetry.
- **Plots**: forest plot (weight-scaled markers, pooled diamond, prediction
  interval) and funnel plot (pseudo-CI cone + Egger annotation).
- **Effect-size intake with provenance** (`convert.py`): recover `(mean, SD)`
  from a reported median with range/IQR (Wan 2014 SD, Luo 2018 mean), and back
  out a standard error from a CI, p-value, t-statistic, or F(1, df). Every
  recovered number carries a `Derived` record of the raw inputs, estimator,
  assumptions, and citation; `to_effects()` threads that lineage into the final
  `(yi, vi)` and `extraction_log()` renders a supplementary extraction table.

## Install

```bash
cd meta_analysis
pip install -e .           # add [pandas] or [test] for extras
```

Or run in place with `PYTHONPATH=. python your_script.py`.

## Quick start

Already have effect sizes and standard errors:

```python
from metanalysis import meta_analyze

res = meta_analyze(yi=[0.10, 0.30, 0.50, -0.05],
                   sei=[0.10, 0.10, 0.12, 0.20],
                   method="REML")           # "FE", "DL", or "REML"
print(res.summary())
res.estimate, res.ci_low, res.ci_high, res.tau2, res.I2, res.pi_low
```

Start from a table of raw study data (any array-like, incl. pandas columns):

```python
import pandas as pd
from metanalysis import compute_effects, meta_analyze, forest_plot, funnel_plot

df = pd.DataFrame({"e1": [15, 8, 20], "n1": [100, 90, 110],
                   "e2": [10, 5, 9],  "n2": [100, 95, 105]})

yi, vi = compute_effects("OR", e1=df.e1, n1=df.n1, e2=df.e2, n2=df.n2)
res = meta_analyze(yi=yi, vi=vi, method="DL")

forest_plot(res, labels=["S1", "S2", "S3"], exp=True).figure.savefig("forest.png")
funnel_plot(res, egger=True).figure.savefig("funnel.png")
```

`compute_effects(measure, ...)` supports `MD`, `SMD`, `OR`, `RR`, `RD`, `COR`.
For ratio measures pool on the log scale and pass `exp=True` to `forest_plot`
to display on the ratio scale.

## Prediction interval

The prediction interval estimates where the true effect of a *new* study is
expected to fall (as opposed to the CI, which bounds the *mean* effect). It is
`estimate ± t(k-2) · sqrt(τ² + SE²)` and is reported for random-effects models
with `k ≥ 3`.

## Validation

The statistics are checked three ways:

- **Exact unit tests** against hand-computed reference values (`tests/`).
- **Machine-precision agreement** with `statsmodels` on the Colditz (1994) BCG
  vaccine trials — effects, `Q`, `I²`, `τ²(DL)`, and FE/RE pooling all match to
  ~1e-13 (`validation/cross_check_statsmodels.py`).
- **REML** reproduces `metafor`'s published BCG values (estimate −0.7145,
  τ² 0.3132).

```bash
pytest -q                                   # 39 tests
python demo.py                              # runs BCG example, writes figures
python validation/cross_check_statsmodels.py
```

## Layout

```
metanalysis/
  pooling.py   meta_analyze(), MetaResult (FE / DL / REML, Q, I², τ², PI, HKSJ)
  effects.py   effect_md/smd/or/rr/rd/cor, compute_effects()
  convert.py   median/IQR→(mean,SD) [Wan/Luo], SE recovery, Derived provenance
  bias.py      egger_test(), EggerResult
  plots.py     forest_plot(), funnel_plot()
tests/         hand-computed unit tests
validation/    external cross-check vs statsmodels
demo.py        end-to-end example on the BCG dataset
```

## References

- Wan, Wang, Liu & Tong (2014). *Estimating the sample mean and standard
  deviation from the sample size, median, range and/or interquartile range.*
- Luo, Wan, Liu & Tong (2018). *Optimally estimating the sample mean from the
  sample size, median, mid-range, and/or mid-quartile range.*
- DerSimonian & Laird (1986). *Meta-analysis in clinical trials.*
- Viechtbauer (2005). *Bias and efficiency of meta-analytic variance
  estimators in the random-effects model.* (REML)
- Higgins, Thompson & Spiegelhalter (2009). *A re-evaluation of random-effects
  meta-analysis.* (prediction interval)
- Egger et al. (1997). *Bias in meta-analysis detected by a simple, graphical
  test.*
- Borenstein et al. (2009). *Introduction to Meta-Analysis.* (effect sizes)
