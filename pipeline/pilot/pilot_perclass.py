"""Per-CELL-TYPE metrics on the shared eval set, per arm (Jiaqi follow-up).

For each arm's selected checkpoint: predict the cohort's external-validation
cells and report per-cell-type support, accuracy (recall), and one-vs-rest F1.
Output: /oscar/scratch/fperalta/pilot_repair/perclass_{cohort}.csv (arm column).
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required: {e}")

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
DEFAULT_TOKEN_DICT = str(pathlib.Path.home() / "data/fperalta/Geneformer/"
                         "geneformer_repo/geneformer/token_dictionary_gc104M.pkl")
ARMS = ["P", "B", "P2BA", "P2BU", "P2DS"]


def make_collator(path):
    try:
        with open(path, "rb") as f:
            return Collator(token_dictionary=pickle.load(f))
    except TypeError:
        return Collator()


def predict(model_dir, ds, batch=64):
    model = BertForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    coll = make_collator(DEFAULT_TOKEN_DICT)
    keep = ds.remove_columns(
        [c for c in ds.column_names if c not in ("input_ids", "length", "label")])
    dl = DataLoader(keep, batch_size=batch, collate_fn=coll)
    preds = []
    with torch.no_grad():
        for b in dl:
            logits = model(input_ids=b["input_ids"].cuda(),
                           attention_mask=b["attention_mask"].cuda()).logits
            preds.append(logits.argmax(-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ild", "crc", "aida"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    c = args.cohort

    labels = json.loads((S / f"{c}_labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}
    ds = load_from_disk(str(S / f"{c}_eval.dataset"))
    n0 = len(ds)
    ds = ds.filter(lambda b: [t in label2id for t in b["cell_type"]], batched=True)
    ds = ds.map(lambda b: {"label": [label2id[t] for t in b["cell_type"]]},
                batched=True)
    print(f"{c}: {len(ds):,}/{n0:,} eval cells in label space, "
          f"{len(set(ds['cell_type']))} cell types present", flush=True)
    y = np.asarray(ds["label"])

    rows = []
    for arm in ARMS:
        run = S / "runs" / c / arm
        sel = json.loads((run / "selected.json").read_text())["checkpoint"]
        yhat = predict(str(run / "checkpoints" / sel), ds, batch=args.batch)
        for lab_id in sorted(set(y)):
            m = y == lab_id
            rows.append({
                "arm": arm, "cell_type": labels[lab_id],
                "n_cells": int(m.sum()),
                "accuracy": float((yhat[m] == lab_id).mean()),
                "f1": float(f1_score(y == lab_id, yhat == lab_id)),
            })
        worst = min((r for r in rows if r["arm"] == arm), key=lambda r: r["accuracy"])
        print(f"{c} {arm}: worst cell type = {worst['cell_type']} "
              f"acc {worst['accuracy']:.3f} (n={worst['n_cells']})", flush=True)

    out = S / f"perclass_{c}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
