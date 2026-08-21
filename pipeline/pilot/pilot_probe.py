"""Linear-probe cell-type separability on frozen scFM representations.

Jiaqi's ask (2026-08-20): replace the silhouette score with a *trained
classifier* that separates cell types in the representation, to test for
catastrophic forgetting.

For each cohort and each model (pretrained backbone, P, B, P2BA), embed the
external-validation cells (mean-pooled final hidden state) and fit a
logistic-regression probe on a stratified train/test split:

  - cell-type accuracy + macro-F1              -> forgetting signal
      (compare fine-tuned/repaired reps vs the pretrained backbone; a drop
       means fine-tuning erased linearly-decodable cell-type structure)
  - per-class recall                            -> WHICH cell types are lost
  - per-ancestry worst-group macro-F1 under a  -> Kang et al. 2020 decoupling
    CLASS-BALANCED probe                           test: if a fresh balanced
      classifier on P's FROZEN representation recovers worst-group F1, the
      demographic bias lived in the task head, not the representation.

Probe = sklearn LogisticRegression (no GPU); GPU only for embedding.
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required: {e}")

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
DEFAULT_TOKEN_DICT = str(pathlib.Path.home() / "data/fperalta/Geneformer/"
                         "geneformer_repo/geneformer/token_dictionary_gc104M.pkl")
ARMS = ["P", "B", "P2BA"]


def make_collator(path):
    try:
        with open(path, "rb") as f:
            return Collator(token_dictionary=pickle.load(f))
    except TypeError:
        return Collator()


def embed(model_dir, ds, batch=16, token_dict=DEFAULT_TOKEN_DICT):
    # take ONLY the final encoder layer via model.bert(...).last_hidden_state --
    # output_hidden_states=True materializes all 25 layers and OOMs a shared GPU
    model = BertForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    coll = make_collator(token_dict)
    keep = ds.remove_columns(
        [c for c in ds.column_names if c not in ("input_ids", "length", "label")])
    dl = DataLoader(keep, batch_size=batch, collate_fn=coll)
    out = []
    with torch.no_grad():
        for b in dl:
            ids, mask = b["input_ids"].cuda(), b["attention_mask"].cuda()
            h = model.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def per_group_worst_f1(groups, y_true, y_pred):
    """Worst-group macro-F1 over cell types common to every group."""
    df = pd.DataFrame({"g": groups, "y": y_true, "p": y_pred})
    sets = [set(sub.y.unique()) for _, sub in df.groupby("g")]
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        common = sorted(df.y.unique())
    per = {g: f1_score(sub.y, sub.p, labels=common, average="macro")
           for g, sub in df.groupby("g")}
    worst = min(per, key=per.get)
    return worst, per[worst], per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ild", "crc", "aida"])
    ap.add_argument("--model-dir", required=True, help="pretrained backbone dir")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-cells", type=int, default=9000)
    ap.add_argument("--test-frac", type=float, default=0.30)
    args = ap.parse_args()
    c = args.cohort

    if (S / f"probe_{c}.csv").exists():
        print(f"{c}: probe_{c}.csv already exists, skipping", flush=True)
        return

    ds = load_from_disk(str(S / f"{c}_eval.dataset"))
    if len(ds) > args.max_cells:
        ds = ds.shuffle(seed=0).select(range(args.max_cells)).flatten_indices()
    ds = ds.map(lambda b: {"label": [0] * len(b["cell_type"])}, batched=True)
    ctypes = np.asarray(ds["cell_type"])
    groups = np.asarray(ds["group"])

    # stratified split needs >=2 cells per class; drop singleton cell types
    vc = pd.Series(ctypes).value_counts()
    keep_types = set(vc[vc >= 2].index)
    if len(keep_types) < len(vc):
        mask = np.array([t in keep_types for t in ctypes])
        print(f"dropping {int((~mask).sum())} cells in "
              f"{len(vc) - len(keep_types)} singleton cell types", flush=True)
        ds = ds.select(np.flatnonzero(mask).tolist())
        ctypes, groups = ctypes[mask], groups[mask]

    print(f"{c}: {len(ds):,} cells, {len(set(ctypes))} cell types, "
          f"{len(set(groups))} groups", flush=True)

    # one stratified split reused for every model, so probe scores are comparable
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=args.test_frac, random_state=0,
                              stratify=ctypes)

    arms = {"pretrained": args.model_dir}
    for a in ARMS:
        run = S / "runs" / c / a
        sel = json.loads((run / "selected.json").read_text())["checkpoint"]
        arms[a] = str(run / "checkpoints" / sel)

    rows, perclass_rows = [], []
    for arm, ckpt in arms.items():
        X = embed(ckpt, ds, batch=args.batch)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = ctypes[tr], ctypes[te]
        gte = groups[te]

        # plain probe: cell-type decodability (forgetting)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
        yp = clf.predict(Xte)
        acc = accuracy_score(yte, yp)
        mf1 = f1_score(yte, yp, average="macro")

        # class-balanced probe: Kang decoupling test on the frozen rep
        clf_b = LogisticRegression(max_iter=2000, n_jobs=-1,
                                   class_weight="balanced").fit(Xtr, ytr)
        yp_b = clf_b.predict(Xte)
        wg_name, wg_f1, per_g = per_group_worst_f1(gte, yte, yp_b)

        rows.append({"cohort": c, "arm": arm, "probe_celltype_acc": acc,
                     "probe_celltype_macro_f1": mf1,
                     "balanced_probe_worst_group": wg_name,
                     "balanced_probe_worst_group_f1": wg_f1,
                     "n_train": len(tr), "n_test": len(te)})
        print(rows[-1], flush=True)
        print(f"  {arm} per-group balanced-probe F1: "
              f"{ {g: round(v, 3) for g, v in per_g.items()} }", flush=True)

        for lab in sorted(set(yte)):
            m = yte == lab
            perclass_rows.append({"cohort": c, "arm": arm, "cell_type": lab,
                                  "n": int(m.sum()),
                                  "probe_recall": float((yp[m] == lab).mean())})

    pd.DataFrame(rows).to_csv(S / f"probe_{c}.csv", index=False)
    pd.DataFrame(perclass_rows).to_csv(S / f"probe_perclass_{c}.csv", index=False)
    print(f"WROTE probe_{c}.csv + probe_perclass_{c}.csv", flush=True)


if __name__ == "__main__":
    main()
