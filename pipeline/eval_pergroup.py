"""Per-group evaluation + honest model selection.

Modes:
  select : score EVERY epoch checkpoint on the run's donor-held validation set;
           pick the one with the best WORST-CLASS accuracy (LaBonte rule).
  final  : evaluate ONE checkpoint on a metadata-bearing dataset (eval_donor
           holdout or an external set); per-group macro-F1 + worst-group +
           donor-level stratified bootstrap CIs.
"""
import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required: {e}")


DEFAULT_TOKEN_DICT = str(pathlib.Path.home() / "data/fperalta/Geneformer/"
                         "geneformer_repo/geneformer/token_dictionary_gc104M.pkl")


def make_collator(token_dict_path):
    """Newer geneformer requires token_dictionary=; older takes no kwarg."""
    try:
        with open(token_dict_path, "rb") as f:
            td = pickle.load(f)
        return Collator(token_dictionary=td)
    except TypeError:
        return Collator()


def common_label_set(df):
    """Classes present in EVERY group's y_true -- the only set on which
    per-group macro-F1 numbers are commensurable for ranking groups."""
    sets = [set(sub.label.unique()) for _, sub in df.groupby("group")]
    common = sorted(set.intersection(*sets)) if sets else []
    return common or sorted(df.label.unique())


def predict(model_dir, ds, batch=64, token_dict=None):
    model = BertForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    coll = make_collator(token_dict or DEFAULT_TOKEN_DICT)
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


def pergroup_table(df):
    common = common_label_set(df)
    rows = []
    for g, sub in df.groupby("group"):
        rows.append({"group": g, "n_cells": len(sub),
                     "n_donors": sub.donor_key.nunique(),
                     "accuracy": accuracy_score(sub.label, sub.pred),
                     "macro_f1": f1_score(sub.label, sub.pred, average="macro"),
                     # fixed common-class set: the number groups are RANKED on
                     "macro_f1_common": f1_score(sub.label, sub.pred,
                                                 labels=common, average="macro")})
    t = pd.DataFrame(rows).sort_values("macro_f1_common")
    return t


def worst_class_acc(df):
    per_class = df.groupby("label").apply(lambda s: accuracy_score(s.label, s.pred))
    return float(per_class.min())


def donor_bootstrap(df, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    common = common_label_set(df)
    out = {}
    for g, sub in df.groupby("group"):
        donors = sub.donor_key.unique()
        by_donor = {d: s for d, s in sub.groupby("donor_key")}
        stats = []
        for _ in range(n_boot):
            pick = rng.choice(donors, size=len(donors), replace=True)
            boot = pd.concat([by_donor[d] for d in pick])
            stats.append(f1_score(boot.label, boot.pred,
                                  labels=common, average="macro"))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        out[g] = (float(lo), float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "final"])
    ap.add_argument("--run-dir", required=True, help="finetune output dir")
    ap.add_argument("--dataset", help="final mode: metadata-bearing dataset dir")
    ap.add_argument("--checkpoint", help="final mode: explicit checkpoint path")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-val-cells", type=int, default=60000,
                    help="select mode: subsample the validation set to this "
                         "many cells for checkpoint scoring (0 = all; full-set "
                         "scoring costs hours per checkpoint on big corpora)")
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    cfg = json.loads((run / "run_config.json").read_text())

    if args.mode == "select":
        val = load_from_disk(str(run / "val_with_meta"))
        if args.max_val_cells and len(val) > args.max_val_cells:
            val = val.shuffle(seed=0).select(range(args.max_val_cells))
            val = val.flatten_indices()
            print(f"selection val subsampled to {len(val):,} cells", flush=True)
        meta = pd.DataFrame({"group": val["group"],
                             "donor_key": val["donor_key"],
                             "label": val["label"]})
        results = []
        for ck in sorted((run / "checkpoints").glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1])):
            if not (ck / "config.json").exists():
                print(f"skipping incomplete checkpoint dir {ck.name}", flush=True)
                continue
            df = meta.copy()
            df["pred"] = predict(str(ck), val, args.batch,
                                 token_dict=cfg.get("token_dict"))
            res = {"checkpoint": ck.name,
                   "overall_macro_f1": f1_score(df.label, df.pred, average="macro"),
                   "worst_class_acc": worst_class_acc(df),
                   "worst_group_f1": pergroup_table(df).macro_f1_common.min()}
            results.append(res)
            print(res, flush=True)
        if not results:
            raise SystemExit("no complete checkpoints found under checkpoints/")
        tab = pd.DataFrame(results)
        tab.to_csv(run / "selection_table.csv", index=False)
        # worst-class acc first; overall macro-F1 breaks ties (all-zero
        # worst-class ties are near-certain with rare classes)
        best = tab.sort_values(["worst_class_acc", "overall_macro_f1"]).iloc[-1]
        (run / "selected.json").write_text(json.dumps(
            {"checkpoint": best.checkpoint,
             "rule": "max worst_class_acc on donor-held validation; "
                     "ties broken by overall macro-F1"}, indent=2))
        print(f"SELECTED {best.checkpoint} (worst-class acc "
              f"{best.worst_class_acc:.4f})", flush=True)
    else:
        ck = args.checkpoint or (run / "checkpoints" /
                                 json.loads((run / "selected.json").read_text())["checkpoint"])
        ds = load_from_disk(args.dataset)
        # the full tokenized corpus carries train+holdout together: keep ONLY
        # the eval_donor holdout, or every number is leakage-inflated.
        # External sets (no 'split' column) pass through untouched.
        if "split" in ds.column_names:
            n0 = len(ds)
            ds = ds.filter(lambda b: [s == "eval_donor" for s in b["split"]],
                           batched=True)
            print(f"holdout filter: kept {len(ds):,}/{n0:,} eval_donor cells",
                  flush=True)
            if len(ds) == 0:
                raise SystemExit("dataset has a 'split' column but no "
                                 "eval_donor cells -- wrong dataset?")
        # align the eval set to the run's label space: map cell_type -> coarse ->
        # label id with the run's own map; drop cells outside the trained space
        labels = cfg["labels"]
        label2id = {l: i for i, l in enumerate(labels)}
        coarse = pd.read_csv(pathlib.Path(cfg["coarse_map"]))
        c2l = dict(zip(coarse.cell_type, coarse.coarse_label))
        ds = ds.map(lambda b: {"coarse": [c2l.get(c, "DROP") for c in b["cell_type"]]},
                    batched=True)
        n0 = len(ds)
        ds = ds.filter(lambda b: [c in label2id for c in b["coarse"]], batched=True)
        if len(ds) < n0:
            print(f"note: dropped {n0 - len(ds):,}/{n0:,} eval cells outside "
                  f"the trained label space", flush=True)
        ds = ds.map(lambda b: {"label": [label2id[c] for c in b["coarse"]]},
                    batched=True)
        df = pd.DataFrame({"group": ds["group"], "donor_key": ds["donor_key"],
                           "label": ds["label"]})
        df["pred"] = predict(str(ck), ds, args.batch,
                             token_dict=cfg.get("token_dict"))

        # Per-cell predictions. The aggregate tables cannot be re-derived into
        # donor bootstraps, confusion matrices, or per-group x per-cell-type
        # breakdowns, and re-deriving them needs the checkpoint -- which stays on
        # scratch and disappears with cluster access. ~1MB per arm; dump it once.
        preds = df.copy()
        preds["cell_type"] = ds["cell_type"]
        preds["label_name"] = [labels[i] for i in preds.label]
        preds["pred_name"] = [labels[i] for i in preds.pred]
        preds.to_csv(run / "final_eval_predictions.csv", index=False)
        print(f"wrote per-cell predictions: {len(preds):,} rows", flush=True)

        table = pergroup_table(df)
        cis = donor_bootstrap(df, args.bootstrap)
        table["ci_low"] = table.group.map(lambda g: cis[g][0])
        table["ci_high"] = table.group.map(lambda g: cis[g][1])
        outp = run / f"final_eval_{pathlib.Path(args.dataset).name}.csv"
        table.to_csv(outp, index=False)
        print(table.to_string(index=False), flush=True)
        print(f"worst group: {table.iloc[0].group} "
              f"(common-class macro-F1 {table.iloc[0].macro_f1_common:.4f} "
              f"CI [{table.iloc[0].ci_low:.4f},{table.iloc[0].ci_high:.4f}])", flush=True)


if __name__ == "__main__":
    main()
