"""Donor-bootstrap CIs and confusion structure from the dumped per-cell predictions.

Runs offline on `results/pilot_repair`, no GPU and no cluster. This is the whole
point of dumping predictions: the rare-cell-type claim rests on single-digit cell
counts (AIDA n=7 going 0.000 -> 0.571 is four cells out of seven), so "did P2BJ
fix it" cannot be answered by a point estimate.

Two things it computes:

  1. Per (model, cohort, arm, cell_type) accuracy with a donor-bootstrap 95% CI.
     Donors are the resampling unit, not cells -- cells within a donor are
     correlated, so resampling cells would give CIs that are far too narrow.

  2. PAIRED deltas between arms. Both arms score the identical eval cells, so the
     same donor resample is applied to both and the difference is computed within
     each resample. That is what makes "P2BJ beats P by 0.57 [0.21, 0.86]" a
     statement about the arms rather than about donor sampling. An unpaired
     comparison of two marginal CIs would be strictly weaker and would overstate
     the uncertainty.

Usage:
    python pipeline/analyze_predictions.py --results-root results/pilot_repair
    python pipeline/analyze_predictions.py --confusion --cohort aida --model geneformer
"""
import argparse
import pathlib
import re

import numpy as np
import pandas as pd

# where each model's predictions live, relative to the results root
SOURCES = [
    ("geneformer",           "runs/*/*/final_eval_predictions.csv"),
    ("scgpt",                "runs_scgpt_fixed/*/*/final_eval_predictions.csv"),
    ("scfoundation_frozen",  "runs_scfoundation_aligned/*/final_eval_predictions.csv"),
    ("scfoundation_ft",      "runs_scfa_ft/*/*/final_eval_predictions.csv"),
]
ARMS = ["P", "B", "P2BA", "P2BJ", "P2BU", "P2DS"]


def load_all(root):
    """One tidy frame: model, cohort, arm, donor_key, cell_type, label, pred."""
    frames = []
    for model, pat in SOURCES:
        for p in sorted(pathlib.Path(root).glob(pat)):
            parts = p.parts
            # .../runs_x/<cohort>/<arm>/file  or  .../runs_x/<cohort>/file
            cohort = parts[-3] if p.parent.name in ARMS else parts[-2]
            df = pd.read_csv(p)
            df["model"] = model
            df["cohort"] = cohort
            # the frozen scFoundation file stacks all arms and carries its own
            # `arm` column; the others are one file per arm
            if "arm" not in df.columns:
                df["arm"] = p.parent.name
            frames.append(df)
    if not frames:
        raise SystemExit(f"no final_eval_predictions.csv under {root}")
    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(out):,} predicted cells: "
          f"{out.model.nunique()} models x {out.cohort.nunique()} cohorts x "
          f"{out.arm.nunique()} arms", flush=True)
    return out


def _multiplicities(donors, n_boot, rng):
    """(n_donors x n_boot) matrix of how many times each donor is drawn.

    The bootstrap is expressed as counts rather than as index arrays so the whole
    resample becomes a matrix product (see _acc_matrix). Building 2000 explicit
    index arrays per file and boolean-masking each one was ~100x slower and made
    the full 57-file run impractical.
    """
    uniq, codes = np.unique(donors, return_inverse=True)
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    M = np.zeros((len(uniq), n_boot), dtype=np.float64)
    for b in range(n_boot):
        np.add.at(M[:, b], picks[b], 1.0)
    return uniq, codes, M


def _acc_matrix(codes, n_donors, ct_codes, n_types, correct, M):
    """Bootstrap accuracy per (cell type x resample), vectorised.

    For cell type t and donor d let C[t,d] = correct cells and N[t,d] = total
    cells. A resample with donor multiplicities m gives
        acc[t] = (C @ m) / (N @ m)
    which is exact -- resampling a donor k times counts its cells k times -- and
    reduces the entire bootstrap to two matrix products.
    """
    flat = ct_codes * n_donors + codes
    N = np.bincount(flat, minlength=n_types * n_donors).reshape(n_types, n_donors)
    C = np.bincount(flat, weights=correct.astype(np.float64),
                    minlength=n_types * n_donors).reshape(n_types, n_donors)
    num, den = C @ M, N @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
    return acc


def per_type_ci(sub, n_boot, seed=0):
    """Accuracy + donor-bootstrap CI for every cell type in one arm."""
    rng = np.random.default_rng(seed)
    donors = sub.donor_key.to_numpy()
    y, p = sub.label.to_numpy(), sub.pred.to_numpy()
    ct = sub.cell_type.to_numpy()
    uniq_d, codes, M = _multiplicities(donors, n_boot, rng)
    types, ct_codes = np.unique(ct, return_inverse=True)
    acc = _acc_matrix(codes, len(uniq_d), ct_codes, len(types), p == y, M)
    rows = []
    for i, t in enumerate(types):
        m = ct == t
        vals = acc[i][~np.isnan(acc[i])]
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size
                  else (np.nan, np.nan))
        rows.append({"cell_type": t, "n_cells": int(m.sum()),
                     "n_donors": int(pd.Series(donors[m]).nunique()),
                     "accuracy": float((p[m] == y[m]).mean()),
                     "ci_low": float(lo), "ci_high": float(hi),
                     "n_boot_used": int(vals.size)})
    return pd.DataFrame(rows)


def paired_delta(df, model, cohort, arm_a, arm_b, n_boot, seed=0):
    """CI on (arm_b - arm_a) per cell type, sharing each donor resample.

    Returns NaN CIs for types absent from either arm's eval frame.
    """
    a = df[(df.model == model) & (df.cohort == cohort) & (df.arm == arm_a)]
    b = df[(df.model == model) & (df.cohort == cohort) & (df.arm == arm_b)]
    if a.empty or b.empty:
        return pd.DataFrame()
    # both arms score the same eval set in the same order; verify before pairing
    if len(a) != len(b) or not (a.label.to_numpy() == b.label.to_numpy()).all():
        raise SystemExit(f"{model}/{cohort}: {arm_a} and {arm_b} eval frames "
                         "differ -- cannot pair")
    rng = np.random.default_rng(seed)
    donors = a.donor_key.to_numpy()
    y = a.label.to_numpy()
    pa, pb = a.pred.to_numpy(), b.pred.to_numpy()
    ct = a.cell_type.to_numpy()
    # ONE multiplicity matrix drives both arms -- that is what makes it paired
    uniq_d, codes, M = _multiplicities(donors, n_boot, rng)
    types, ct_codes = np.unique(ct, return_inverse=True)
    nd, nt = len(uniq_d), len(types)
    acc_a = _acc_matrix(codes, nd, ct_codes, nt, pa == y, M)
    acc_b = _acc_matrix(codes, nd, ct_codes, nt, pb == y, M)
    diff = acc_b - acc_a
    rows = []
    for i, t in enumerate(types):
        m = ct == t
        vals = diff[i][~np.isnan(diff[i])]
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size
                  else (np.nan, np.nan))
        rows.append({"model": model, "cohort": cohort, "cell_type": t,
                     "n_cells": int(m.sum()),
                     f"acc_{arm_a}": float((pa[m] == y[m]).mean()),
                     f"acc_{arm_b}": float((pb[m] == y[m]).mean()),
                     "delta": float((pb[m] == y[m]).mean()
                                    - (pa[m] == y[m]).mean()),
                     "ci_low": float(lo), "ci_high": float(hi),
                     "excludes_zero": bool(vals.size and (lo > 0 or hi < 0))})
    return pd.DataFrame(rows)


def confusion(df, model, cohort, arm, top=12):
    """What the failures actually get called instead."""
    s = df[(df.model == model) & (df.cohort == cohort) & (df.arm == arm)]
    if s.empty:
        return pd.DataFrame()
    wrong = s[s.label != s.pred]
    return (wrong.groupby(["cell_type", "pred_name"]).size()
            .reset_index(name="n").sort_values("n", ascending=False).head(top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/pilot_repair")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rare-max", type=int, default=100,
                    help="cell types at or below this count are 'rare'")
    ap.add_argument("--baseline", default="P")
    ap.add_argument("--compare", default="P2BJ")
    ap.add_argument("--confusion", action="store_true")
    ap.add_argument("--model", default="geneformer")
    ap.add_argument("--cohort", default="aida")
    args = ap.parse_args()

    df = load_all(args.results_root)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.confusion:
        c = confusion(df, args.model, args.cohort, args.compare)
        print(f"\n=== {args.model} / {args.cohort} / {args.compare}: "
              f"most common confusions ===")
        print(c.to_string(index=False) if len(c) else "(no data)")
        return

    # 1. marginal CIs everywhere
    per = []
    for (m, c, a), sub in df.groupby(["model", "cohort", "arm"]):
        t = per_type_ci(sub, args.n_boot)
        t.insert(0, "arm", a); t.insert(0, "cohort", c); t.insert(0, "model", m)
        per.append(t)
    per = pd.concat(per, ignore_index=True)
    per.to_csv(out / "perclass_with_ci.csv", index=False)
    print(f"wrote {out/'perclass_with_ci.csv'} ({len(per):,} rows)")

    # 2. the comparison the claim actually rests on
    deltas = []
    for (m, c) in df.groupby(["model", "cohort"]).groups:
        d = paired_delta(df, m, c, args.baseline, args.compare, args.n_boot)
        if len(d):
            deltas.append(d)
    if deltas:
        d = pd.concat(deltas, ignore_index=True)
        d.to_csv(out / f"delta_{args.baseline}_vs_{args.compare}.csv", index=False)
        print(f"wrote {out/f'delta_{args.baseline}_vs_{args.compare}.csv'}")

        rare = d[d.n_cells <= args.rare_max].sort_values("delta", ascending=False)
        print(f"\n=== rare cell types (n <= {args.rare_max}): "
              f"{args.compare} - {args.baseline} ===")
        print(rare[["model", "cohort", "cell_type", "n_cells",
                    f"acc_{args.baseline}", f"acc_{args.compare}",
                    "delta", "ci_low", "ci_high", "excludes_zero"]]
              .to_string(index=False, max_colwidth=26))
        sig = int(rare.excludes_zero.sum())
        print(f"\n{sig}/{len(rare)} rare-type gains have a 95% CI excluding zero.")
        print("Those are the ones that survive as claims; the rest are "
              "consistent with donor sampling noise.")


if __name__ == "__main__":
    main()
