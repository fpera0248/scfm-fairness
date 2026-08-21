"""DFR / cRT test: does retraining ONLY the classifier head on balanced data
(frozen backbone) recover worst-group F1 as well as a full repair fine-tune?

The probe showed the frozen representation carries the balanced information; this
tests the deployable version WITHOUT the in-distribution shortcut — the head is
trained on the balanced TRAINING cells and evaluated on the held-out external
set, exactly like the models' own heads (fig8), so the numbers are comparable.

Per cohort, per backbone in {pretrained, P}:
  - embed the balanced-augmented TRAINING cells (ba) and the external-validation
    cells with the FROZEN backbone
  - train a linear head (logistic regression) on the ba features:
       balanced  = DFR / cRT (class-balanced last layer)
  - for the P backbone, also train a PLAIN head on the imbalanced (prop) features
    as a control that should reproduce the biased deployed head
  - report per-ancestry worst-group macro-F1 (common classes) on the eval set

Compare against fig8 own-head worst-group: CRC P .83 / B .86 / P2BA .86;
ILD P .71 / B .80 / P2BA .81; AIDA P .38 / B .54 / P2BA .60.
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


def make_collator(path):
    try:
        with open(path, "rb") as f:
            return Collator(token_dictionary=pickle.load(f))
    except TypeError:
        return Collator()


def embed(model_dir, ds, batch=24, token_dict=DEFAULT_TOKEN_DICT):
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


def worst_group(df):
    """Per-ancestry macro-F1 over cell types common to every group; return
    (worst_group, worst_f1, per_group_dict) -- matches eval_pergroup / fig8."""
    sets = [set(sub.y.unique()) for _, sub in df.groupby("g")]
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        common = sorted(df.y.unique())
    per = {g: f1_score(sub.y, sub.p, labels=common, average="macro")
           for g, sub in df.groupby("g")}
    w = min(per, key=per.get)
    return w, per[w], per


def load_xy(ds, tag):
    a = load_from_disk(str(S / f"{ds}_{tag}.dataset"))
    a = a.map(lambda b: {"label": [0] * len(b["cell_type"])}, batched=True)
    return a, np.asarray(a["cell_type"]), np.asarray(a["group"])


def eval_head(clf, l2i, Xe, ye, ge):
    keep = np.array([t in l2i for t in ye])
    yv = np.array([l2i[t] for t in ye[keep]])
    pv = clf.predict(Xe[keep])
    df = pd.DataFrame({"g": ge[keep], "y": yv, "p": pv})
    return worst_group(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["ild", "crc", "aida"])
    ap.add_argument("--model-dir", required=True, help="pretrained backbone dir")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    c = args.cohort

    ba, y_ba, _ = load_xy(c, "ba")
    prop, y_pr, _ = load_xy(c, "prop")
    ev, y_ev, g_ev = load_xy(c, "eval")
    print(f"{c}: ba={len(ba):,} prop={len(prop):,} eval={len(ev):,}", flush=True)

    backbones = {"pretrained": args.model_dir}
    psel = json.loads((S / "runs" / c / "P" / "selected.json").read_text())["checkpoint"]
    backbones["P"] = str(S / "runs" / c / "P" / "checkpoints" / psel)

    rows = []
    for bname, bdir in backbones.items():
        Xba = embed(bdir, ba, args.batch)
        Xev = embed(bdir, ev, args.batch)
        labels = sorted(set(y_ba))
        l2i = {l: i for i, l in enumerate(labels)}
        yba = np.array([l2i[t] for t in y_ba])

        # DFR / cRT: class-balanced linear head on the balanced training features
        dfr = LogisticRegression(max_iter=2000, n_jobs=-1,
                                 class_weight="balanced").fit(Xba, yba)
        w, wf1, per = eval_head(dfr, l2i, Xev, y_ev, g_ev)
        rows.append({"cohort": c, "backbone": bname, "head": "DFR_balanced_on_ba",
                     "worst_group": w, "worst_group_f1": wf1})
        print(rows[-1], {g: round(v, 3) for g, v in per.items()}, flush=True)

        if bname == "P":
            # control: plain head on the imbalanced (prop) features -> should
            # reproduce the biased deployed head (bias comes from the head)
            Xpr = embed(bdir, prop, args.batch)
            labels_p = sorted(set(y_pr))
            l2ip = {l: i for i, l in enumerate(labels_p)}
            ypr = np.array([l2ip[t] for t in y_pr])
            plain = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xpr, ypr)
            w2, wf2, per2 = eval_head(plain, l2ip, Xev, y_ev, g_ev)
            rows.append({"cohort": c, "backbone": bname,
                         "head": "plain_on_prop_control",
                         "worst_group": w2, "worst_group_f1": wf2})
            print(rows[-1], {g: round(v, 3) for g, v in per2.items()}, flush=True)

    pd.DataFrame(rows).to_csv(S / f"dfr_{c}.csv", index=False)
    print(f"WROTE dfr_{c}.csv", flush=True)


if __name__ == "__main__":
    main()
