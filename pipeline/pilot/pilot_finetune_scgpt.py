"""scGPT version of the repair pilot: fine-tune scGPT (whole-human) on ONE pilot
condition, optionally initialized from a prior fine-tuned run (--init-from), then
select by worst-class val accuracy (LaBonte) and final-eval per-ancestry
worst-group macro-F1 on the shared external-validation set.

Same data, arms, label space, and metrics as the Geneformer pilot — Jiaqi's
"same pipeline, code and data on other foundation models" request.

Gene-space note: pilot h5ads carry Ensembl IDs; scGPT's vocab is gene symbols.
We invert Geneformer's ensembl_mapping_dict (symbol->ENSG) to map ENSG->symbol.
"""
import argparse
import json
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import torch
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
ROOT = pathlib.Path.home() / "data/fperalta/scfoundation"
GF_ASSETS = pathlib.Path.home() / "data/fperalta/Geneformer/geneformer_repo/geneformer"
PAD_TOKEN, PAD_VALUE, MAX_LEN = "<pad>", -2, 3001

ILD = ROOT / "augmentedv4/ethnicity_scfoundation_workflow"
CRC = ROOT / "augmented_CRC/ethnicity_scfoundation_workflow"
AIDA = ROOT / "augmented_AIDA/ethnicity_scfoundation_workflow"
FILES = {
    "ild": {"prop": ILD / "ILD_Ethnicity_Pilot_Proportional_2497_ETHNICITY.h5ad",
            "ba": ILD / "ILD_Ethnicity_Pilot_BalancedAugmented_2143Each_ETHNICITY.h5ad",
            "bu": ILD / "ILD_Ethnicity_Pilot_BalancedUpsampled_2143Each_ETHNICITY.h5ad",
            "ds": ILD / "ILD_Ethnicity_Pilot_Downsampled_48Each_ETHNICITY.h5ad",
            "eval": ILD / "ILD_Ethnicity_External_Validation_12500.h5ad"},
    "crc": {"prop": CRC / "CRC_Eth_Pilot_Proportional_2497_ETHNICITY.h5ad",
            "ba": CRC / "CRC_Eth_Pilot_BalancedAugmented_1880Each_ETHNICITY.h5ad",
            "bu": CRC / "CRC_Eth_Pilot_BalancedUpsampled_1880Each_ETHNICITY.h5ad",
            "ds": CRC / "CRC_Eth_Pilot_Downsampled_48Each_ETHNICITY.h5ad",
            "eval": CRC / "CRC_Eth_External_Validation_8572.h5ad"},
    "aida": {"prop": AIDA / "AIDA_Ethnicity_Pilot_Proportional_2500_ETHNICITY.h5ad",
             "ba": AIDA / "AIDA_Ethnicity_Pilot_BalancedAugmented_779Each_ETHNICITY.h5ad",
             "bu": AIDA / "AIDA_Ethnicity_Pilot_BalancedUpsampled_779Each_ETHNICITY.h5ad",
             "ds": AIDA / "AIDA_Ethnicity_Pilot_Downsampled_92Each_ETHNICITY.h5ad",
             "eval": AIDA / "AIDA_Ethnicity_External_Validation_12500.h5ad"},
}


def ens2sym_map():
    with open(GF_ASSETS / "ensembl_mapping_dict_gc104M.pkl", "rb") as f:
        sym2ens = pickle.load(f)
    out = {}
    for sym, ens in sym2ens.items():
        out.setdefault(ens, sym)
    return out


def load_pilot(path, tag, vocab, e2s):
    a = ad.read_h5ad(path)
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    a.obs["source"] = src
    grp = a.obs["self_reported_ethnicity"].astype(str).str.strip().str.lower()
    a.obs["group"] = np.where(src == "synthetic", "synthetic", grp)
    a.obs["donor_key"] = np.where(src == "synthetic", "synthetic",
                                  a.obs["donor_id"].astype(str))
    if tag == "eval":
        keep = ~a.obs["group"].isin(["synthetic", "unknown", "nan", ""])
        a = a[keep.to_numpy()].copy()
    # ENSG -> symbol -> keep genes in scGPT vocab
    syms = np.array([e2s.get(g, "") for g in a.var_names])
    in_vocab = np.array([s in vocab and s != "" for s in syms])
    a = a[:, in_vocab].copy()
    a.var["gene_name"] = syms[in_vocab]
    a.var_names = pd.Index(a.var["gene_name"]).astype(str)
    a.var_names_make_unique()
    return a


def preprocess(a):
    pre = Preprocessor(use_key="X", filter_gene_by_counts=False,
                       filter_cell_by_counts=False, normalize_total=1e4,
                       result_normed_key="X_normed", log1p=True,
                       result_log1p_key="X_log1p", subset_hvg=False,
                       binning=51, result_binned_key="X_binned")
    pre(a, batch_key=None)
    return a


def tokenize(a, vocab):
    genes = a.var_names.tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    counts = a.layers["X_binned"]
    counts = counts.A if hasattr(counts, "A") else np.asarray(counts)
    return tokenize_and_pad_batch(counts, gene_ids, max_len=MAX_LEN, vocab=vocab,
                                  pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
                                  append_cls=True, include_zero_gene=False)


def common_label_set(df):
    sets = [set(sub.label.unique()) for _, sub in df.groupby("group")]
    common = sorted(set.intersection(*sets)) if sets else []
    return common or sorted(df.label.unique())


def predict(model, tok, batch, device):
    model.eval()
    ds = TensorDataset(tok["genes"], tok["values"])
    preds = []
    with torch.no_grad():
        for g, v in DataLoader(ds, batch_size=batch):
            g, v = g.to(device), v.to(device).float()
            mask = g.eq(model.vocab[PAD_TOKEN]) if hasattr(model, "vocab") else g.eq(0)
            out = model(g, v, src_key_padding_mask=mask, CLS=True)
            preds.append(out["cls_output"].argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(FILES))
    ap.add_argument("--cond", required=True, choices=["prop", "ba", "bu", "ds"])
    ap.add_argument("--init-from", default=None,
                    help="stage-1 outdir; loads its best_model.pt")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-dir",
                    default=str(pathlib.Path.home() / "data/fperalta/fresh_scGPT/scGPT_human"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--freeze-layers", type=int, default=6,
                    help="bottom encoder layers to freeze (whole-human has 12)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-donor-frac", type=float, default=0.15)
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = pathlib.Path(args.model_dir)

    vocab = GeneVocab.from_file(mdir / "vocab.json")
    for tok in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])
    e2s = ens2sym_map()

    c = args.cohort
    labels = json.loads((S / f"{c}_labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    train_a = preprocess(load_pilot(FILES[c][args.cond], args.cond, vocab, e2s))
    eval_a = preprocess(load_pilot(FILES[c]["eval"], "eval", vocab, e2s))
    for a, nm in [(train_a, "train"), (eval_a, "eval")]:
        keep = a.obs["cell_type"].astype(str).isin(label2id).to_numpy()
        print(f"{nm}: keep {keep.sum():,}/{a.n_obs:,} in label space", flush=True)
        if keep.sum() < a.n_obs:
            a._inplace_subset_obs(keep)
    y_all = np.array([label2id[t] for t in train_a.obs["cell_type"].astype(str)])
    y_eval = np.array([label2id[t] for t in eval_a.obs["cell_type"].astype(str)])

    # donor-held validation from REAL donors (fallback: random 10%)
    real = sorted(set(train_a.obs.loc[train_a.obs.source == "real", "donor_key"]))
    if len(real) >= 4:
        n_val = max(1, int(len(real) * args.val_donor_frac))
        vd = set(rng.choice(real, size=n_val, replace=False))
        is_val = (train_a.obs["donor_key"].isin(vd) & (train_a.obs["source"] == "real")).to_numpy()
        is_trn = ~train_a.obs["donor_key"].isin(vd).to_numpy()
    else:
        is_val = rng.random(train_a.n_obs) < 0.10
        is_trn = ~is_val
    print(f"train={is_trn.sum():,} val={is_val.sum():,}", flush=True)

    tok_all = tokenize(train_a, vocab)
    tok_eval = tokenize(eval_a, vocab)
    g_t, v_t = tok_all["genes"][is_trn], tok_all["values"][is_trn]
    y_t = torch.tensor(y_all[is_trn], dtype=torch.long)
    g_v, v_v = tok_all["genes"][is_val], tok_all["values"][is_val]
    y_v = y_all[is_val]

    margs = json.loads((mdir / "args.json").read_text())
    model = TransformerModel(
        ntoken=len(vocab), d_model=margs.get("embsize", 512),
        nhead=margs.get("nheads", 8), d_hid=margs.get("d_hid", 512),
        nlayers=margs.get("nlayers", 12), nlayers_cls=3, n_cls=len(labels),
        vocab=vocab, dropout=0.2, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
        do_mvc=False, do_dab=False, use_batch_labels=False,
        domain_spec_batchnorm=False, input_emb_style="continuous",
        n_input_bins=margs.get("n_bins", 51) + 2, cell_emb_style="cls",
        use_fast_transformer=False)
    src_file = (pathlib.Path(args.init_from) / "best_model.pt"
                if args.init_from else mdir / "best_model.pt")
    pre_sd = torch.load(src_file, map_location="cpu")
    msd = model.state_dict()
    loadable = {k: v for k, v in pre_sd.items()
                if k in msd and v.shape == msd[k].shape}
    msd.update(loadable)
    model.load_state_dict(msd)
    print(f"loaded {len(loadable)}/{len(msd)} tensors from {src_file}", flush=True)
    if args.freeze_layers > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for layer in model.transformer_encoder.layers[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    model.to(device)

    (out / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "n_labels": len(labels), "labels": labels}, indent=2))

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()
    trn = DataLoader(TensorDataset(g_t, v_t, y_t), batch_size=args.batch,
                     shuffle=True)
    pad_id = vocab[PAD_TOKEN]
    best = {"worst": -1, "epoch": -1}
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for g, v, y in trn:
            g, v, y = g.to(device), v.to(device).float(), y.to(device)
            out_d = model(g, v, src_key_padding_mask=g.eq(pad_id), CLS=True)
            loss = lossf(out_d["cls_output"], y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss)
        yhat = predict(model, {"genes": g_v, "values": v_v}, args.batch, device)
        per = pd.Series(y_v).groupby(y_v).apply(
            lambda s: (yhat[s.index] == s.values).mean())
        res = {"epoch": ep, "loss": tot / max(len(trn), 1),
               "val_acc": float(accuracy_score(y_v, yhat)),
               "val_macro_f1": float(f1_score(y_v, yhat, average="macro")),
               "val_worst_class": float(per.min()),
               "secs": round(time.time() - t0)}
        print(res, flush=True)
        score = (res["val_worst_class"], res["val_macro_f1"])
        if score > (best["worst"], best.get("f1", -1)):
            best = {"worst": res["val_worst_class"], "f1": res["val_macro_f1"],
                    "epoch": ep}
            torch.save(model.state_dict(), out / "best_model.pt")
    (out / "selected.json").write_text(json.dumps(best, indent=2))
    print(f"SELECTED epoch {best['epoch']}", flush=True)

    # final eval on external set with the selected weights
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    yhat = predict(model, tok_eval, args.batch, device)
    df = pd.DataFrame({"group": eval_a.obs["group"].to_numpy(),
                       "donor_key": eval_a.obs["donor_key"].to_numpy(),
                       "label": y_eval, "pred": yhat})
    common = common_label_set(df)
    rows = [{"group": g, "n_cells": len(sub),
             "macro_f1_common": f1_score(sub.label, sub.pred, labels=common,
                                         average="macro")}
            for g, sub in df.groupby("group")]
    t = pd.DataFrame(rows).sort_values("macro_f1_common")
    t.to_csv(out / "final_eval_pergroup.csv", index=False)
    pc = [{"cell_type": labels[l], "n_cells": int((y_eval == l).sum()),
           "accuracy": float((yhat[y_eval == l] == l).mean())}
          for l in sorted(set(y_eval))]
    pd.DataFrame(pc).to_csv(out / "final_eval_perclass.csv", index=False)
    print(t.to_string(index=False), flush=True)
    print(f"worst group: {t.iloc[0].group} ({t.iloc[0].macro_f1_common:.4f})",
          flush=True)
    print("SCGPT ARM COMPLETE", flush=True)


if __name__ == "__main__":
    main()
