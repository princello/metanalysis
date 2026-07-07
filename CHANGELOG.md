# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning once it reaches 1.0.

## [Unreleased]

### Added
- **Effect-size intake layer** (`metanalysis/convert.py`): recover `(mean, SD)`
  from a reported median with range/IQR (Wan 2014 SD, Luo 2018 optimal weighted
  mean), and recover a standard error from a confidence interval, two-sided
  p-value, t-statistic, or `F(1, df)`. Every recovered value is a `Derived`
  record carrying its raw inputs, estimator, assumptions, and citation;
  `to_effects()` threads that lineage into the final `(yi, vi)` and
  `extraction_log()` renders a supplementary extraction table.
- **Hartung–Knapp–Sidik–Jonkman (HKSJ) adjustment** to the random-effects
  confidence interval via `meta_analyze(..., test="knha")`; the default
  `test="z"` behavior is unchanged. Pinned against metafor `rma(test="knha")`.
- **Packaging & project metadata**: MIT `LICENSE`, full `pyproject.toml`
  metadata (authors, license, keywords, classifiers, URLs), a `MANIFEST.in` so
  the sdist ships `demo.py` and `validation/`, and a GitHub Actions CI workflow
  running the suite and the statsmodels cross-check on Python 3.9–3.13.

### Fixed
- Guarded six degenerate-input paths that previously crashed or returned silent
  `NaN`/nonsense: HKSJ with zero residual heterogeneity (`q=0`); non-finite
  `yi`/`vi`; confidence `level` outside `(0, 1)` (in `meta_analyze` and
  `funnel_plot`); `effect_cor` with `n <= 3` or `|r| >= 1`; `effect_smd` with
  `n1 + n2 <= 2`; and Egger's test when all standard errors are equal. Each now
  raises a clear `ValueError`.
- `validation/cross_check_statsmodels.py` now runs from the repo root without
  requiring the package to be installed or `PYTHONPATH` to be set.

### Documentation
- Documented the conventions that intentionally differ from `metafor` (Q-based
  REML I²/H², the Higgins–Thompson–Spiegelhalter prediction interval, the
  HKSJ/PI SE mismatch, the Borenstein SMD variance, and the classic `lm` Egger
  variant), both in-code and in a README "Conventions & differences from
  metafor" section.

## [0.1.0]

### Added
- Core pooling (`meta_analyze`, `MetaResult`): fixed-effect inverse-variance,
  DerSimonian–Laird, and REML random-effects models; Cochran's Q, I², H², τ²,
  and a prediction interval. Validated against `statsmodels` (~1e-13) and
  `metafor` (REML BCG values).
- Effect sizes (`effects.py`): raw and standardized mean difference, log OR/RR,
  risk difference, and Fisher's z correlation, with zero-cell continuity
  correction.
- Egger's regression test for funnel-plot asymmetry (`bias.py`).
- Forest and funnel plots (`plots.py`).
