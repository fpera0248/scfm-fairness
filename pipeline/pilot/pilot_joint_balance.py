"""Track B: ANCESTRY x CELL-TYPE jointly balanced repair set.

Jiaqi's critique (2026-08-25): the repair arms barely move rare cell types.
Diagnosis: our BalancedAugmented pilots equalise cells per ANCESTRY only
(`..._779Each_ETHNICITY`); WITHIN each ancestry the cell-type distribution is
still long-tailed, so nothing ever asked the model to fix cell-type rarity.

This builds a training set that equalises the (ancestry x cell_type) GRID by
resampling the existing augmented pool, then tokenizes it as {cohort}_bj.dataset
so pilot_finetune.py can train arm P2BJ exactly like the other repair arms.

Honest limit: a stratum with ZERO real+synthetic cells cannot be filled by
resampling — those are reported, not silently skipped.
"""
import argparse
import collections
import json
import os
import pathlib
import shutil

import anndata as ad
import numpy as np
import pandas as pd

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
ROOT = pathlib.Path.home() / "data/fperalta/scfoundation"
ATTRS = ["cell_type", "group", "split", "donor_key", "source", "condition"]

BA = {
    "ild": ROOT / "augmentedv4/ethnicity_scfoundation_workflow/"
                  "ILD_Ethnicity_Pilot_BalancedAugmented_2143Each_ETHNICITY.h5ad",
    "crc": ROOT / "augmented_CRC/ethnicity_scfoundation_workflow/"
                  "CRC_Eth_Pilot_BalancedAugmented_1880Each_ETHNICITY.h5ad",
    "aida": ROOT / "augmented_AIDA/ethnicity_scfoundation_workflow/"
                   "AIDA_Ethnicity_Pilot_BalancedAugmented_779Each_ETHNICITY.h5ad",
}


def make_tokenizer(nproc=4):
    # imported lazily so --h5ad-only can regenerate the joint-balanced h5ad for
    # scGPT / scFoundation from an env that has no Geneformer installed
    from geneformer import TranscriptomeTokenizer
    d = pathlib.Path.home() / "data/fperalta/Geneformer/geneformer_repo/geneformer"
    kw = {}
    if (d / "token_dictionary_gc104M.pkl").exists():
        kw = {"token_dictionary_file": str(d / "token_dictionary_gc104M.pkl"),
              "gene_median_file": str(d / "gene_median_dictionary_gc104M.pkl"),
              "gene_mapping_file": str(d / "ensembl_mapping_dict_gc104M.pkl")}
    try:
        return TranscriptomeTokenizer({a: a for a in ATTRS}, nproc=nproc,
                                      model_version="V2", **kw)
    except TypeError:
        return TranscriptomeTokenizer({a: a for a in ATTRS}, nproc=nproc, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(BA))
    ap.add_argument("--per-stratum", type=int, default=0,
                    help="cells per (ancestry x cell_type) cell; 0 = median non-empty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h5ad-only", action="store_true",
                    help="write the h5ad and stop; the Geneformer .dataset already "
                         "exists and re-tokenizing it would only burn time")
    args = ap.parse_args()
    c, rng = args.cohort, np.random.default_rng(args.seed)

    a = ad.read_h5ad(BA[c])
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    a.obs["source"] = src
    grp = a.obs["self_reported_ethnicity"].astype(str).str.strip().str.lower()
    # synthetic cells inherit the ancestry they were generated for
    a.obs["group"] = grp.to_numpy()
    a.obs["donor_key"] = np.where(src == "synthetic", "synthetic",
                                  a.obs["donor_id"].astype(str))
    a.obs["split"] = "train"
    a.obs["condition"] = "bj"
    ct = a.obs["cell_type"].astype(str).to_numpy()
    gp = a.obs["group"].to_numpy()

    groups = sorted(set(gp) - {"nan", "unknown", ""})
    types = sorted(set(ct))
    counts = collections.Counter(zip(gp, ct))
    nonzero = [counts[(g, t)] for g in groups for t in types if counts[(g, t)] > 0]
    K = args.per_stratum or max(20, int(np.median(nonzero)))
    empty = [(g, t) for g in groups for t in types if counts[(g, t)] == 0]
    print(f"{c}: {len(groups)} ancestries x {len(types)} cell types = "
          f"{len(groups)*len(types)} strata; {len(empty)} EMPTY (cannot be filled "
          f"by resampling); target {K} cells/stratum", flush=True)
    print(f"  stratum sizes: min {min(nonzero)} median {int(np.median(nonzero))} "
          f"max {max(nonzero)}", flush=True)

    idx = []
    for g in groups:
        for t in types:
            pool = np.flatnonzero((gp == g) & (ct == t))
            if pool.size == 0:
                continue
            take = rng.choice(pool, size=K, replace=pool.size < K)
            idx.extend(take.tolist())
    rng.shuffle(idx)
    print(f"  built {len(idx):,} cells "
          f"({len(idx)//max(len(groups),1):,} per ancestry, balanced within each)",
          flush=True)

    b = a[np.array(idx)].copy()
    b.obs_names_make_unique()
    b.obs["ensembl_ok"] = True
    b.var["ensembl_id"] = (b.var["feature_id"].to_numpy()
                           if "feature_id" in b.var.columns else b.var_names.to_numpy())
    b.obs["n_counts"] = np.asarray(b.X.sum(axis=1)).ravel()
    for col in ATTRS:
        b.obs[col] = b.obs[col].astype(str)

    # report the grid we actually achieved
    got = pd.crosstab(b.obs["group"], b.obs["cell_type"])
    (S / f"{c}_bj_grid.csv").write_text(got.to_csv())
    print(f"  grid written: {got.shape[0]} x {got.shape[1]}, "
          f"per-cell min {got.values[got.values>0].min()} max {got.values.max()}",
          flush=True)

    # keep the h5ad: Geneformer trains from the tokenized copy, but scGPT and
    # scFoundation read h5ad directly, and P2BJ has to exist for all three models
    # or it cannot appear in a cross-model comparison
    # Write via a pid-unique temp + atomic rename. The scGPT and scFoundation
    # drivers both build this on demand and can reach the same cohort at once;
    # a half-written h5ad passes their `[ -f ]` check and then fails to load.
    keep = S / f"{c}_bj.h5ad"
    tmp = S / f".{c}_bj.h5ad.{os.getpid()}.tmp"
    b.write_h5ad(tmp)
    os.replace(tmp, keep)
    print(f"  h5ad kept for the non-Geneformer models: {keep}", flush=True)
    if args.h5ad_only:
        print(f"H5AD ONLY: {keep}", flush=True)
        return

    stage = S / f"stage_{c}_bj"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    b.write_h5ad(stage / f"{c}_bj.h5ad")
    del a, b
    out = S / f"{c}_bj.dataset"
    if out.exists():
        shutil.rmtree(out)
    make_tokenizer().tokenize_data(str(stage), str(S), f"{c}_bj", file_format="h5ad")
    shutil.rmtree(stage)
    print(f"JOINT-BALANCED SET READY: {out}", flush=True)


if __name__ == "__main__":
    main()
