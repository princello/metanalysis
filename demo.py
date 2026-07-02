"""Demonstration of the metanalysis toolkit on the BCG vaccine trials.

Runs the full workflow: raw 2x2 counts -> log risk ratios -> fixed-effect,
DerSimonian-Laird, and REML pooling -> Egger's test -> forest and funnel
plots. Figures are written next to this script.

Run: python demo.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metanalysis import (
    compute_effects,
    egger_test,
    forest_plot,
    funnel_plot,
    meta_analyze,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Colditz et al. (1994) BCG tuberculosis-vaccine trials.
bcg = pd.DataFrame({
    "trial": ["Martin 1980", "Ferguson 1949", "Rosenthal 1960", "Hart 1977",
              "Frimodt-M. 1973", "Stein 1953", "Vandiviere 1973", "Madras 1980",
              "Coetzee 1968", "Rosenthal 1961", "Comstock 1974",
              "Comstock 1969", "Comstock 1976"],
    "tpos": [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27],
    "tneg": [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699,
             50448, 2493, 16886],
    "cpos": [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29],
    "cneg": [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600,
             27197, 2338, 17825],
})

# Effect sizes: log risk ratio, vaccinated vs control.
yi, vi = compute_effects(
    "RR",
    e1=bcg["tpos"], n1=bcg["tpos"] + bcg["tneg"],
    e2=bcg["cpos"], n2=bcg["cpos"] + bcg["cneg"],
)

print("=" * 66)
for method in ("FE", "DL", "REML"):
    print(meta_analyze(yi=yi, vi=vi, method=method).summary())
    print("-" * 66)

print(egger_test(yi=yi, vi=vi).summary())
print("=" * 66)

# Figures.
res = meta_analyze(yi=yi, vi=vi, method="REML")

ax = forest_plot(res, labels=list(bcg["trial"]), exp=True,
                 xlabel="Risk ratio (vaccine vs control)")
ax.figure.savefig(os.path.join(HERE, "forest.png"), dpi=300,
                  bbox_inches="tight", facecolor="white")
ax.figure.savefig(os.path.join(HERE, "forest.pdf"),
                  bbox_inches="tight", facecolor="white")
plt.close(ax.figure)

ax = funnel_plot(res, egger=True, title="Funnel plot (log risk ratio)")
ax.figure.savefig(os.path.join(HERE, "funnel.png"), dpi=300,
                  bbox_inches="tight", facecolor="white")
ax.figure.savefig(os.path.join(HERE, "funnel.pdf"),
                  bbox_inches="tight", facecolor="white")
plt.close(ax.figure)

print("Wrote forest.png/.pdf and funnel.png/.pdf to", HERE)
