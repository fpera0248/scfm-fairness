"""fig29: the direct answer to "maybe check results from other foundation models".

Jiaqi circled the zero-accuracy points on the corrected scFoundation figure and
said "for these cell types, all model fail". This takes exactly those cell types
and shows every model on every arm, so the question is answered on its own terms.

fig27 does NOT do this. It filters to cases where the imbalanced baseline scored
zero, which drops models that already handled a circled type -- for CD141+ myeloid
dendritic and plasma cell it shows only 1 of 4 models. That filtering answers
"where did the baseline fail", not "what do the other models do here".

The circled types are read from the scFoundation-frozen predictions rather than
hardcoded, so this stays honest if the underlying run is ever regenerated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

RES = pathlib.Path("/Users/fperaltacastro/scfm-fairness/results/pilot_repair")
COHORT = "aida"
ARMS = ["P", "B", "P2BA", "P2BJ", "P2BU", "P2DS"]
MODEL_COLOR = {"geneformer": "#3B3A6E", "scgpt": "#4C9C8E",
               "scfoundation_ft": "#E1743B", "scfoundation_frozen": "#FBC79A"}
MODEL_LABEL = {"geneformer": "Geneformer", "scgpt": "scGPT",
               "scfoundation_ft": "scFoundation (fine-tuned)",
               "scfoundation_frozen": "scFoundation (frozen)"}
SHORT = {"CD141-positive myeloid dendritic cell": "CD141+ myeloid dendritic",
         "pre-conventional dendritic cell": "pre-conventional dendritic",
         "double negative T regulatory cell": "double negative T regulatory",
         "hematopoietic stem cell": "hematopoietic stem cell",
         "plasma cell": "plasma cell"}


def load(model, pat, stacked=False):
    """-> {(arm, cell_type): accuracy}, plus per-type n."""
    acc, n = {}, {}
    if stacked:
        p = RES / pat
        if not p.exists():
            return acc, n
        df = pd.read_csv(p)
        for (a, t), s in df.groupby(["arm", "cell_type"]):
            acc[(a, t)] = (s.label == s.pred).mean()
            n[t] = len(s)
    else:
        for a in ARMS:
            p = RES / pat.format(arm=a)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            for t, s in df.groupby("cell_type"):
                acc[(a, t)] = (s.label == s.pred).mean()
                n[t] = len(s)
    return acc, n


DATA, NCELLS = {}, {}
for model, pat, stacked in [
        ("geneformer", f"runs/{COHORT}/{{arm}}/final_eval_predictions.csv", False),
        ("scgpt", f"runs_scgpt_fixed/{COHORT}/{{arm}}/final_eval_predictions.csv", False),
        ("scfoundation_ft", f"runs_scfa_ft/{COHORT}/{{arm}}/final_eval_predictions.csv", False),
        ("scfoundation_frozen", f"runs_scfoundation_aligned/{COHORT}/final_eval_predictions.csv", True)]:
    a, n = load(model, pat, stacked)
    DATA[model] = a
    NCELLS.update(n)

# the circled types: any cell type scFoundation-frozen scores 0.000 on in any arm
frozen = DATA["scfoundation_frozen"]
circled = sorted({t for (a, t), v in frozen.items() if v == 0},
                 key=lambda t: NCELLS[t])
print("circled types:", [(NCELLS[t], t) for t in circled])

fig, axes = plt.subplots(1, len(circled), figsize=(15.2, 4.5), sharey=True)
W = 0.20
for ax, t in zip(axes, circled):
    for mi, model in enumerate(MODEL_LABEL):
        xs, ys = [], []
        for ai, a in enumerate(ARMS):
            v = DATA[model].get((a, t))
            if v is None:
                continue
            xs.append(ai + (mi - 1.5) * W)
            ys.append(v)
        ax.bar(xs, ys, width=W, color=MODEL_COLOR[model], edgecolor="black",
               linewidth=0.55, zorder=3,
               label=MODEL_LABEL[model] if t == circled[0] else None)
        # a zero bar is invisible; mark it so "still fails" stays readable
        for x, y in zip(xs, ys):
            if y == 0:
                ax.plot([x], [0.012], marker="x", ms=4.4, mew=1.5,
                        color=MODEL_COLOR[model], zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS, fontsize=8, rotation=45, ha="right")
    ax.set_title(f"{SHORT.get(t, t)}\n(n = {NCELLS[t]})", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # neutral highlight: a warm tint would compete with the scFoundation hues
    ax.axvspan(2.5, 3.5, color="#EDEDED", zorder=0)   # highlight P2BJ

axes[0].set_ylabel("accuracy", fontsize=10)
fig.legend(loc="lower center", ncol=4, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.015))
fig.suptitle("All four models on the cell types flagged as failing (AIDA)\n"
             "shaded = joint balancing (P2BJ);   × = accuracy 0.000",
             fontsize=12)
fig.tight_layout(rect=[0, 0.075, 1, 0.90])
out = "/Users/fperaltacastro/Downloads/bibm_figures/fig29_circled_types_all_models.png"
fig.savefig(out, dpi=300)
print("saved", out)

for t in circled:
    row = [f"{DATA[m].get(('P2BJ', t), float('nan')):.3f}" for m in MODEL_LABEL]
    base = [f"{DATA[m].get(('P', t), float('nan')):.3f}" for m in MODEL_LABEL]
    print(f"  n={NCELLS[t]:3} {t[:34]:34} P={','.join(base)}  P2BJ={','.join(row)}")
