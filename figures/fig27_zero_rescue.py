"""fig27: the direct answer to "for these cell types, all model fail".

Every case where the imbalanced baseline scores exactly 0.000, with what joint
ancestry x cell-type balancing (P2BJ) does to it, and a donor-bootstrap CI on the
change. Grouped by cell type so multi-model replication is visible at a glance --
that is the actual claim, and a per-model panel would hide it.

Palette is the established model system (scFoundation orange / Geneformer navy /
scGPT teal), with the two scFoundation adaptations distinguished by fill rather
than hue so the model identity stays readable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLOR = {"geneformer": "#3B3A6E", "scgpt": "#4C9C8E",
               "scfoundation_ft": "#E1743B", "scfoundation_frozen": "#FBC79A"}
MODEL_LABEL = {"geneformer": "Geneformer", "scgpt": "scGPT",
               "scfoundation_ft": "scFoundation (fine-tuned)",
               "scfoundation_frozen": "scFoundation (frozen)"}
HATCH = {}

d = pd.read_csv("/Users/fperaltacastro/scfm-fairness/results/delta_P_vs_P2BJ.csv")
z = d[(d.n_cells <= 100) & (d.acc_P == 0.0)].copy()
SHORT = {"CD141-positive myeloid dendritic cell": "CD141+ myeloid dendritic",
         "pre-conventional dendritic cell": "pre-conventional dendritic",
         "double negative T regulatory cell": "double negative T regulatory",
         "granzyme K-associated CD8 T cell": "granzyme K CD8 T",
         "plasmacytoid dendritic cell": "plasmacytoid dendritic"}
z["short"] = z.cell_type.map(lambda s: SHORT.get(s, s))
z["key"] = z.cohort.str.upper() + "  " + z["short"] + \
           "  (n=" + z.n_cells.astype(str) + ")"
# order groups by best achieved rescue, so the strongest evidence reads first
order = (z.groupby("key").acc_P2BJ.max().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(11.4, 7.4))
ypos, yticks, ylabels = 0.0, [], []
BAR, GAP = 0.62, 0.85
for key in order:
    g = z[z.key == key].sort_values("acc_P2BJ", ascending=False)
    ys = []
    for _, r in g.iterrows():
        c = MODEL_COLOR[r.model]
        if r.acc_P2BJ == 0:
            # A zero-height bar is invisible, which would silently hide the models
            # that were tested and STILL fail -- the honest half of the result.
            # Mark them explicitly instead.
            ax.plot([0], [ypos], marker="x", ms=7, mew=2.0, color=c, zorder=5)
            ax.text(0.028, ypos, "still 0.000", fontsize=7.4, va="center",
                    color="#444444", style="italic", zorder=5)
        else:
            ax.barh(ypos, r.acc_P2BJ, height=BAR, color=c, edgecolor="black",
                    linewidth=0.7, hatch=HATCH.get(r.model, ""), zorder=3)
            # CI on the CHANGE; baseline is 0 so the bar end is also the delta
            ax.plot([r.ci_low, r.ci_high], [ypos, ypos], color="black", lw=1.4,
                    zorder=4, solid_capstyle="butt")
            for xb in (r.ci_low, r.ci_high):
                ax.plot([xb, xb], [ypos - .17, ypos + .17], color="black",
                        lw=1.4, zorder=4)
            if r.excludes_zero:
                ax.text(r.ci_high + 0.022, ypos, "*", fontsize=13, va="center",
                        fontweight="bold", zorder=5)
        ys.append(ypos)
        ypos += GAP
    yticks.append(np.mean(ys))
    ylabels.append(key)
    ypos += 0.55

ax.axvline(0, color="black", lw=1.1, zorder=2)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=8.6)
ax.invert_yaxis()
ax.set_xlim(-0.08, 1.20)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("accuracy after joint balancing   (baseline = 0.000; "
              "bars show 95% CI)", fontsize=10)
ax.grid(axis="x", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR[m], ec="black", lw=0.7,
                         hatch=HATCH.get(m, "")) for m in MODEL_LABEL]
handles += [plt.Line2D([], [], color="black", marker="*", linestyle="None",
                       markersize=9),
            plt.Line2D([], [], color="#555555", marker="x", linestyle="None",
                       markersize=7, mew=2.0)]
ax.legend(handles,
          list(MODEL_LABEL.values()) + ["CI excludes zero", "still 0.000"],
          fontsize=8.4, loc="lower right", framealpha=0.96)
ax.set_title("Joint balancing rescues 18 of 21 cell types the baseline never "
             "predicts\n10 with a 95% CI excluding zero",
             fontsize=12, pad=12)
fig.tight_layout()
out = "/Users/fperaltacastro/Downloads/bibm_figures/fig27_zero_rescue_ci.png"
fig.savefig(out, dpi=300)
print("saved", out)
print(f"  {len(z)} zero-baseline cases, {(z.acc_P2BJ>0).sum()} rescued, "
      f"{z.excludes_zero.sum()} significant")
