"""fig28: where joint balancing helps, across every cell type and all four models.

fig27 answers the specific question about zero-accuracy types. This is the wider
claim: the benefit is concentrated at low cell counts and fades to nothing on
abundant types, which is what you would expect if P2BJ is fixing a distributional
problem rather than just moving overall accuracy around.

Points are filled when the donor-bootstrap CI excludes zero, hollow otherwise --
so the eye is not asked to trust an effect that the interval does not support.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLOR = {"geneformer": "#3B3A6E", "scgpt": "#4C9C8E",
               "scfoundation_ft": "#E1743B", "scfoundation_frozen": "#B4551F"}
MODEL_LABEL = {"geneformer": "Geneformer", "scgpt": "scGPT",
               "scfoundation_ft": "scFoundation (fine-tuned)",
               "scfoundation_frozen": "scFoundation (frozen)"}

d = pd.read_csv("/Users/fperaltacastro/scfm-fairness/results/delta_P_vs_P2BJ.csv")
fig, ax = plt.subplots(figsize=(11.0, 6.0))
ax.axhline(0, color="black", lw=1.1, zorder=2)
ax.axvspan(0.8, 100, color="#F2F2F2", zorder=0)

for m, g in d.groupby("model"):
    sig, ns = g[g.excludes_zero], g[~g.excludes_zero]
    ax.scatter(ns.n_cells, ns.delta, s=34, facecolors="none",
               edgecolors=MODEL_COLOR[m], lw=1.1, alpha=0.75, zorder=3)
    ax.scatter(sig.n_cells, sig.delta, s=46, c=MODEL_COLOR[m],
               edgecolors="black", lw=0.6, zorder=4, label=MODEL_LABEL[m])

ax.set_xscale("log")
ax.set_xlim(1.4, 6000)
ax.set_ylim(-0.55, 1.15)
ax.set_xlabel("cells of that type in the evaluation set (log scale)", fontsize=10)
ax.set_ylabel("accuracy change: P2BJ − imbalanced baseline", fontsize=10)
ax.grid(alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

# The trade-off is the honest headline: significant GAINS and significant LOSSES
# counted separately per abundance band. A single "N significant" figure would
# hide that the losses are concentrated in abundant classes.
lines = []
for lo, hi, lab in [(0, 100, "rare (n ≤ 100)"),
                    (101, 1000, "mid (101–1000)"),
                    (1001, 10**9, "abundant (> 1000)")]:
    s = d[(d.n_cells >= lo) & (d.n_cells <= hi)]
    up = (s.excludes_zero & (s.delta > 0)).sum()
    dn = (s.excludes_zero & (s.delta < 0)).sum()
    lines.append(f"{lab:18} +{up:3} sig gain   −{dn:3} sig loss   "
                 f"median {s.delta.median():+.3f}")
ax.text(0.015, 0.035, "\n".join(lines), transform=ax.transAxes, fontsize=8.4,
        va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.7))
ax.text(0.055, 0.955, "shaded: rare cell types", transform=ax.transAxes,
        fontsize=8.4, color="#555555", va="top")

h, l = ax.get_legend_handles_labels()
h.append(plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="none",
                    markeredgecolor="#555555", markersize=7))
l.append("CI includes zero (not significant)")
ax.legend(h, l, fontsize=8.5, loc="upper right", framealpha=0.96, ncol=1)
ax.set_title("Joint balancing trades abundant-class accuracy for rare-class "
             "accuracy\n"
             "gains concentrate below n≈100; losses concentrate above n≈1000. "
             "4 model variants × 3 cohorts, donor-bootstrap 95% CIs",
             fontsize=11.5, pad=11)
fig.tight_layout()
out = "/Users/fperaltacastro/Downloads/bibm_figures/fig28_delta_vs_abundance.png"
fig.savefig(out, dpi=300)
print("saved", out)
for line in lines:
    print("  " + line)
print(f"  net: {(d.excludes_zero & (d.delta>0)).sum()} sig gains vs "
      f"{(d.excludes_zero & (d.delta<0)).sum()} sig losses over {len(d)} types")
