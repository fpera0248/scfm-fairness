"""Tokenize the BIBM corpus for Geneformer V2 (ILD collection excluded).

Per part: filter to selected non-ILD cells, attach arm/group/donor metadata,
add ensembl_id + n_counts, tokenize with TranscriptomeTokenizer (V2 dict),
save one arrow dataset per part (resumable). Final: concatenate + save.
"""
import os
import pathlib
import shutil

import anndata as ad
import numpy as np
import pandas as pd

from geneformer import TranscriptomeTokenizer
from datasets import load_from_disk, concatenate_datasets

CORPUS = pathlib.Path("/oscar/scratch/fperalta/bibm_corpus")
TOKD = CORPUS / "tokenized_v2_parts"
FINAL = CORPUS / "tokenized_v2"
STAGE = CORPUS / "tok_stage"
ILD_COLLECTION = "07e12576-b41b-4350-adfc-059bfc4328ea"  # Natri et al. ILD — EXCLUDED from training
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS") or 8)

ATTRS = ["joinid", "cell_type", "group", "split", "arm_balanced", "arm_matched",
         "arm_proportional", "donor_key", "dataset_id", "tissue_general",
         "self_reported_ethnicity", "assay"]

TOKD.mkdir(exist_ok=True)
STAGE.mkdir(exist_ok=True)

sel = pd.read_parquet(CORPUS / "selection.parquet")
n0 = len(sel)
sel = sel[sel.collection_id != ILD_COLLECTION]
print(f"selection: {n0:,} cells -> {len(sel):,} after ILD exclusion "
      f"({n0 - len(sel):,} ILD cells dropped)", flush=True)
meta = sel.set_index("soma_joinid")[
    ["group", "split", "arm_balanced", "arm_matched", "arm_proportional",
     "donor_key", "collection_id"]]
keep = set(meta.index)

# token dictionary: prefer the local Geneformer repo assets used by the original
# step2a runs; fall back to the installed package defaults.
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
    tk = TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                nproc=NPROC, model_version="V2", **tk_kwargs)
except TypeError:  # older package without model_version kwarg
    tk = TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                nproc=NPROC, **tk_kwargs)
    print("NOTE: package lacks model_version kwarg — verify dictionary vintage matches V2",
          flush=True)

parts = sorted((CORPUS / "h5ad_parts").glob("part_*.h5ad"))
print(f"{len(parts)} parts to process", flush=True)
for p in parts:
    out = TOKD / f"{p.stem}.dataset"
    marker = TOKD / f"{p.stem}.empty"
    if out.exists() or marker.exists():
        continue
    a = ad.read_h5ad(p)
    jid = a.obs["soma_joinid"].astype(np.int64)
    mask = jid.isin(keep).to_numpy()
    if not mask.any():
        marker.touch()
        print(f"{p.stem}: 0 kept (all excluded)", flush=True)
        continue
    a = a[mask].copy()
    m = meta.loc[a.obs["soma_joinid"].astype(np.int64).to_numpy()]
    a.obs["joinid"] = a.obs["soma_joinid"].astype(np.int64).to_numpy()
    a.obs["group"] = m["group"].to_numpy()
    a.obs["split"] = m["split"].to_numpy()
    for arm in ("arm_balanced", "arm_matched", "arm_proportional"):
        a.obs[arm] = m[arm].to_numpy().astype(int)
    a.obs["donor_key"] = m["donor_key"].to_numpy()
    a.obs["ensembl_ok"] = True
    a.var["ensembl_id"] = (a.var["feature_id"].to_numpy()
                           if "feature_id" in a.var.columns else a.var_names.to_numpy())
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    for col in ("cell_type", "dataset_id", "tissue_general",
                "self_reported_ethnicity", "assay", "group", "split", "donor_key"):
        a.obs[col] = a.obs[col].astype(str)

    pdir = STAGE / p.stem
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    a.write_h5ad(pdir / f"{p.stem}.h5ad")
    del a
    tk.tokenize_data(str(pdir), str(TOKD), p.stem, file_format="h5ad")
    shutil.rmtree(pdir)
    print(f"{p.stem}: tokenized ({int(mask.sum()):,} cells)", flush=True)

print("concatenating parts...", flush=True)
dsets = [load_from_disk(str(d)) for d in sorted(TOKD.glob("part_*.dataset"))]
full = concatenate_datasets(dsets)
if FINAL.exists():
    shutil.rmtree(FINAL)
full.save_to_disk(str(FINAL))
print(f"TOKENIZATION COMPLETE: {len(full):,} cells -> {FINAL}", flush=True)
print(f"expected (from selection, non-ILD): {len(sel):,}", flush=True)
