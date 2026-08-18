"""Integration-style metrics for the repair pilot (Jiaqi's ask #3).

For each cohort and arm checkpoint (P, B, P2BA + the pretrained backbone as a
frozen reference), embed the shared external-validation cells and compute:

  ethnicity iLISI  — kNN inverse-Simpson of ethnicity labels in each cell's
                     neighborhood (k=90), median over cells; higher = ancestries
                     mix better in the embedding (less ancestry signal).
  cell-type ASW    — silhouette of cell-type labels on the same embedding
                     (PCA-50); higher = biology preserved.

Both metrics on the same cells for every arm, so arms are directly comparable.
Embedding = mean over non-padded token positions of the final hidden layer.
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required: {e}")

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
DEFAULT_TOKEN_DICT = str(pathlib.Path.home() / "data/fperalta/Geneformer/"
                         "geneformer_repo/geneformer/token_dictionary_gc104M.pkl")


def make_collator(path):
    try:
        with open(path, "rb") as f:
            return Collator(token_dictionary=pickle.load(f))
    except TypeError:
        return Collator()


def embed(model_dir, ds, batch=48, token_dict=DEFAULT_TOKEN_DICT):
    model = BertForSequenceClassification.from_pretrained(
        model_dir, output_hidden_states=True).cuda().eval()
    coll = make_collator(token_dict)
    keep = ds.remove_columns(
        [c for c in ds.column_names if c not in ("input_ids", "length", "label")])
    dl = DataLoader(keep, batch_size=batch, collate_fn=coll)
    out = []
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].cuda()
            mask = b["attention_mask"].cuda()
            h = model(input_ids=ids, attention_mask=mask).hidden_states[-1]
            m = mask.unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def ilisi(X, labels, k=90):
    """Median per-cell inverse Simpson index of label mixing among k neighbors."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    labs = pd.Categorical(labels).codes
    neigh = labs[idx]
    scores = np.empty(len(labs), dtype=float)
    for i in range(len(labs)):
        _, counts = np.unique(neigh[i], return_counts=True)
        p = counts / counts.sum()
        scores[i] = 1.0 / np.sum(p * p)
    return float(np.median(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ild", "crc", "aida"])
    ap.add_argument("--model-dir", required=True, help="pretrained Geneformer dir")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-cells", type=int, default=8000,
                    help="subsample eval cells for the metric (speed)")
    args = ap.parse_args()

    c = args.cohort
    ds = load_from_disk(str(S / f"{c}_eval.dataset"))
    if len(ds) > args.max_cells:
        ds = ds.shuffle(seed=0).select(range(args.max_cells)).flatten_indices()
    ds = ds.map(lambda b: {"label": [0] * len(b["cell_type"])}, batched=True)
    groups = np.asarray(ds["group"])
    ctypes = np.asarray(ds["cell_type"])
    print(f"{c}: {len(ds):,} eval cells, {len(set(groups))} groups, "
          f"{len(set(ctypes))} cell types", flush=True)

    arms = {"pretrained": args.model_dir}
    for a in ("P", "B", "P2BA"):
        run = S / "runs" / c / a
        sel = json.loads((run / "selected.json").read_text())["checkpoint"]
        arms[a] = str(run / "checkpoints" / sel)

    rows = []
    for arm, ckpt in arms.items():
        X = embed(ckpt, ds, batch=args.batch)
        Xp = PCA(n_components=50, random_state=0).fit_transform(X)
        row = {
            "cohort": c, "arm": arm,
            "ethnicity_ilisi": ilisi(Xp, groups),
            "celltype_asw": float(silhouette_score(Xp, ctypes)),
            "n_cells": len(ds), "n_groups": len(set(groups)),
            "checkpoint": ckpt,
        }
        rows.append(row)
        print(row, flush=True)

    out = S / f"integration_metrics_{c}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
