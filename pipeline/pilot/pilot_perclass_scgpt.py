"""Per-CELL-TYPE accuracy for the scGPT arms (Jiaqi: "check the other models").

Answers whether the cell types that sit at zero accuracy under scFoundation are
also stuck under scGPT, or whether that is model-specific. Mirrors
pilot_perclass.py (Geneformer) so the three models are directly comparable.

Everything about gene space and preprocessing is imported from
pilot_finetune_scgpt rather than re-derived: the HVGs must be the SAME 1200 genes
the arms were trained on, selected from the cohort's proportional file, or the
checkpoints are being evaluated in a gene space they never saw.

Output: /oscar/scratch/fperalta/pilot_repair/perclass_scgpt_{cohort}.csv
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# reuse the training script's data path verbatim -- do not reimplement it
from pilot_finetune_scgpt import (FILES, PAD_TOKEN, PAD_VALUE, S, ens2sym_map,
                                  load_pilot, preprocess, tokenize, predict,
                                  DEFAULT_HVG)

ARMS = ["P", "B", "P2BA", "P2BJ", "P2BU", "P2DS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(FILES))
    ap.add_argument("--runs-root", default=str(S / "runs_scgpt"))
    ap.add_argument("--model-dir",
                    default=str(pathlib.Path.home() /
                                "data/fperalta/scGPT/scGPT_human"))
    ap.add_argument("--n-hvg", type=int, default=DEFAULT_HVG)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None,
                    help="default perclass_scgpt_{cohort}.csv; pass an explicit "
                         "path to keep the pre-mask-fix numbers for comparison")
    args = ap.parse_args()
    c = args.cohort
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = pathlib.Path(args.model_dir)

    vocab = GeneVocab.from_file(mdir / "vocab.json")
    for tok in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab[PAD_TOKEN])
    e2s = ens2sym_map()

    labels = json.loads((S / f"{c}_labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    # SAME gene space the arms were trained in: HVGs from the proportional file
    hvg_src = preprocess(load_pilot(FILES[c]["prop"], "prop", vocab, e2s),
                         n_hvg=args.n_hvg)
    genes = hvg_src.var_names.tolist()
    del hvg_src
    max_len = len(genes) + 1
    print(f"gene space: {len(genes)} HVGs (seq len {max_len})", flush=True)

    eval_a = preprocess(load_pilot(FILES[c]["eval"], "eval", vocab, e2s),
                        gene_subset=genes)
    keep = eval_a.obs["cell_type"].astype(str).isin(label2id).to_numpy()
    print(f"eval: keep {keep.sum():,}/{eval_a.n_obs:,} in label space", flush=True)
    if keep.sum() < eval_a.n_obs:
        eval_a._inplace_subset_obs(keep)
    y = np.array([label2id[t] for t in eval_a.obs["cell_type"].astype(str)])
    tok_eval = tokenize(eval_a, vocab, max_len)
    # carried per cell so the dumped predictions support donor bootstrapping and
    # per-group breakdowns, not just per-cell-type accuracy
    ev_group = eval_a.obs["group"].astype(str).to_numpy()
    ev_donor = (eval_a.obs["donor_key"].astype(str).to_numpy()
                if "donor_key" in eval_a.obs.columns
                else np.full(eval_a.n_obs, "unknown"))
    ev_ct = eval_a.obs["cell_type"].astype(str).to_numpy()

    margs = json.loads((mdir / "args.json").read_text())
    rows = []
    for arm in ARMS:
        ckpt = pathlib.Path(args.runs_root) / c / arm / "best_model.pt"
        if not ckpt.exists():
            print(f"  {arm}: MISSING {ckpt}, skipping", flush=True)
            continue
        model = TransformerModel(
            ntoken=len(vocab), d_model=margs.get("embsize", 512),
            nhead=margs.get("nheads", 8), d_hid=margs.get("d_hid", 512),
            nlayers=margs.get("nlayers", 12), nlayers_cls=3, n_cls=len(labels),
            vocab=vocab, dropout=0.2, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
            do_mvc=False, do_dab=False, use_batch_labels=False,
            domain_spec_batchnorm=False, input_emb_style="continuous",
            n_input_bins=margs.get("n_bins", 51) + 2, cell_emb_style="cls",
            use_fast_transformer=False)
        # arm checkpoints are saved post-remap, so they load directly
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        yhat = predict(model, tok_eval, args.batch, device, vocab[PAD_TOKEN])
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        pd.DataFrame({"group": ev_group, "donor_key": ev_donor,
                      "cell_type": ev_ct, "label": y, "pred": yhat,
                      "label_name": [labels[i] for i in y],
                      "pred_name": [labels[i] for i in yhat]}).to_csv(
            pathlib.Path(args.runs_root) / c / arm / "final_eval_predictions.csv",
            index=False)

        for lab_id in sorted(set(y)):
            m = y == lab_id
            rows.append({
                "arm": arm, "cell_type": labels[lab_id],
                "n_cells": int(m.sum()),
                "accuracy": float((yhat[m] == lab_id).mean()),
                "f1": float(f1_score(y == lab_id, yhat == lab_id)),
            })
        npred = len(set(yhat))
        worst = min((r for r in rows if r["arm"] == arm),
                    key=lambda r: r["accuracy"])
        print(f"{c} {arm}: {npred}/{len(labels)} cell types ever predicted; "
              f"worst = {worst['cell_type']} acc {worst['accuracy']:.3f} "
              f"(n={worst['n_cells']})", flush=True)

    if not rows:
        # writing a 1-byte csv and exiting 0 reads as success in sacct; it is not
        raise SystemExit(f"no arm checkpoints found under {args.runs_root}/{c} "
                         f"(looked for {ARMS}); nothing to evaluate")
    out = pathlib.Path(args.out) if args.out else S / f"perclass_scgpt_{c}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"WROTE {out} ({len(rows)} rows, "
          f"{len({r['arm'] for r in rows})} arms)", flush=True)


if __name__ == "__main__":
    main()
