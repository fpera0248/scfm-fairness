"""Tokenize the ASI pilot datasets for the imbalanced->balanced repair pilot.

3 cohorts (ILD, CRC, AIDA) x (4 training conditions + 1 external validation):
  prop = Proportional (natural imbalance)     ba = BalancedAugmented (scDesign3)
  bu   = BalancedUpsampled (duplication)      ds = Downsampled (real floor)
  eval = the ASI paper's External_Validation set (real cells)

Outputs under /oscar/scratch/fperalta/pilot_repair/:
  {cohort}_{tag}.dataset       tokenized arrow datasets (Geneformer V2 dict)
  {cohort}_labels.json         FIXED label space per cohort (union of the 4
                               training conditions; stage-1/stage-2 heads must
                               share one geometry)
  {cohort}_identity_map.csv    cell_type -> itself; satisfies eval_pergroup's
                               coarse-map contract without coarsening
Also PRINTS donor overlap between each training pilot and the eval set — the
original ASI splits were cell-level, not donor-level; reported, not hidden.
"""
import json
import os
import pathlib
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from geneformer import TranscriptomeTokenizer

ROOT = pathlib.Path.home() / "data/fperalta/scfoundation"
OUT = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK") or 4)

ILD = ROOT / "augmentedv4/ethnicity_scfoundation_workflow"
CRC = ROOT / "augmented_CRC/ethnicity_scfoundation_workflow"
AIDA = ROOT / "augmented_AIDA/ethnicity_scfoundation_workflow"

COHORTS = {
    "ild": {
        "prop": ILD / "ILD_Ethnicity_Pilot_Proportional_2497_ETHNICITY.h5ad",
        "ba": ILD / "ILD_Ethnicity_Pilot_BalancedAugmented_2143Each_ETHNICITY.h5ad",
        "bu": ILD / "ILD_Ethnicity_Pilot_BalancedUpsampled_2143Each_ETHNICITY.h5ad",
        "ds": ILD / "ILD_Ethnicity_Pilot_Downsampled_48Each_ETHNICITY.h5ad",
        "eval": ILD / "ILD_Ethnicity_External_Validation_12500.h5ad",
    },
    "crc": {
        "prop": CRC / "CRC_Eth_Pilot_Proportional_2497_ETHNICITY.h5ad",
        "ba": CRC / "CRC_Eth_Pilot_BalancedAugmented_1880Each_ETHNICITY.h5ad",
        "bu": CRC / "CRC_Eth_Pilot_BalancedUpsampled_1880Each_ETHNICITY.h5ad",
        "ds": CRC / "CRC_Eth_Pilot_Downsampled_48Each_ETHNICITY.h5ad",
        "eval": CRC / "CRC_Eth_External_Validation_8572.h5ad",
    },
    "aida": {
        "prop": AIDA / "AIDA_Ethnicity_Pilot_Proportional_2500_ETHNICITY.h5ad",
        "ba": AIDA / "AIDA_Ethnicity_Pilot_BalancedAugmented_779Each_ETHNICITY.h5ad",
        "bu": AIDA / "AIDA_Ethnicity_Pilot_BalancedUpsampled_779Each_ETHNICITY.h5ad",
        "ds": AIDA / "AIDA_Ethnicity_Pilot_Downsampled_92Each_ETHNICITY.h5ad",
        "eval": AIDA / "AIDA_Ethnicity_External_Validation_12500.h5ad",
    },
}

ATTRS = ["cell_type", "group", "split", "donor_key", "source", "condition"]


def make_tokenizer():
    tk_kwargs = {}
    cand_dirs = [
        pathlib.Path(os.environ.get("GENEFORMER_ASSETS", "/nonexistent")),
        pathlib.Path.home() / "data/fperalta/Geneformer/geneformer_repo/geneformer",
        pathlib.Path.home() / "data/fperalta/geneformer_repo/geneformer",
    ]
    for d in cand_dirs:
        if (d / "token_dictionary_gc104M.pkl").exists():
            tk_kwargs = {
                "token_dictionary_file": str(d / "token_dictionary_gc104M.pkl"),
                "gene_median_file": str(d / "gene_median_dictionary_gc104M.pkl"),
                "gene_mapping_file": str(d / "ensembl_mapping_dict_gc104M.pkl"),
            }
            print(f"using V2 dictionaries from {d}", flush=True)
            break
    else:
        print("using geneformer package default dictionaries", flush=True)
    try:
        return TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                      nproc=NPROC, model_version="V2", **tk_kwargs)
    except TypeError:
        print("NOTE: package lacks model_version kwarg -- verify dictionary "
              "vintage matches V2", flush=True)
        return TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                      nproc=NPROC, **tk_kwargs)


def norm_group(eth, source):
    if source == "synthetic":
        return "synthetic"
    s = str(eth).strip().lower()
    return "unknown" if s in ("nan", "unknown", "", "none") else s


def load_norm(path, tag):
    a = ad.read_h5ad(path)
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    a.obs["source"] = src
    a.obs["group"] = [norm_group(e, s) for e, s in
                      zip(a.obs["self_reported_ethnicity"], src)]
    a.obs["donor_key"] = np.where(src == "synthetic", "synthetic",
                                  a.obs["donor_id"].astype(str))
    a.obs["split"] = "eval_donor" if tag == "eval" else "train"
    a.obs["condition"] = tag
    if tag == "eval":
        keep = ~a.obs["group"].isin(["unknown", "synthetic"])
        if (~keep).sum():
            print(f"  eval: dropping {(~keep).sum():,} unknown/synthetic cells",
                  flush=True)
        a = a[keep.to_numpy()].copy()
    a.var["ensembl_id"] = a.var_names.to_numpy()
    n_ensg = sum(str(v).startswith("ENSG") for v in a.var_names[:100])
    assert n_ensg > 90, f"{path.name}: var index does not look like Ensembl ids"
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    for col in ATTRS:
        a.obs[col] = a.obs[col].astype(str)
    return a


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tk = make_tokenizer()
    for cohort, files in COHORTS.items():
        print(f"=== {cohort} ===", flush=True)
        label_union, eval_donors, overlap_rows = set(), None, []
        adatas = {}
        for tag, path in files.items():
            outds = OUT / f"{cohort}_{tag}.dataset"
            a = load_norm(path, tag)
            adatas[tag] = a.obs[["donor_key", "source", "cell_type", "group"]].copy()
            if tag != "eval":
                label_union |= set(a.obs["cell_type"])
            comp = a.obs.groupby("group").size().to_dict()
            print(f"  {tag}: {a.n_obs:,} cells {comp}", flush=True)
            if outds.exists():
                print(f"  {tag}: dataset exists, skipping tokenization", flush=True)
                del a
                continue
            stage = OUT / f"stage_{cohort}_{tag}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            a.write_h5ad(stage / f"{cohort}_{tag}.h5ad")
            del a
            tk.tokenize_data(str(stage), str(OUT), f"{cohort}_{tag}",
                             file_format="h5ad")
            shutil.rmtree(stage)
            print(f"  {tag}: tokenized -> {outds.name}", flush=True)

        # donor overlap report (ASI splits were cell-level; we do not hide this)
        ev = adatas["eval"]
        eval_donors = set(ev.loc[ev.source == "real", "donor_key"])
        for tag in ("prop", "ba", "bu", "ds"):
            tr = adatas[tag]
            td = set(tr.loc[tr.source == "real", "donor_key"])
            inter = td & eval_donors
            overlap_rows.append({"condition": tag, "train_donors": len(td),
                                 "eval_donors": len(eval_donors),
                                 "shared_donors": len(inter)})
        print(pd.DataFrame(overlap_rows).to_string(index=False), flush=True)

        (OUT / f"{cohort}_labels.json").write_text(
            json.dumps(sorted(label_union), indent=2))
        all_types = sorted(label_union | set(adatas["eval"]["cell_type"]))
        pd.DataFrame({"cell_type": all_types, "coarse_label": all_types}).to_csv(
            OUT / f"{cohort}_identity_map.csv", index=False)
        print(f"  labels: {len(label_union)} train classes; identity map "
              f"{len(all_types)} types", flush=True)
    print("PREP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
