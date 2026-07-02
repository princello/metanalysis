"""metanalysis: a general-purpose meta-analysis toolkit.

Pool study-level effect sizes with fixed-effect or random-effects models,
compute effect sizes from raw study data, test for small-study effects, and
draw forest and funnel plots.
"""

from .bias import EggerResult, egger_test
from .convert import (
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
from .effects import (
    compute_effects,
    effect_cor,
    effect_md,
    effect_or,
    effect_rd,
    effect_rr,
    effect_smd,
)
from .plots import forest_plot, funnel_plot
from .pooling import MetaResult, meta_analyze

__all__ = [
    "MetaResult",
    "meta_analyze",
    "forest_plot",
    "funnel_plot",
    "effect_md",
    "effect_smd",
    "effect_or",
    "effect_rr",
    "effect_rd",
    "effect_cor",
    "compute_effects",
    "EggerResult",
    "egger_test",
    # effect-size intake / conversion layer
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
__version__ = "0.1.0"
