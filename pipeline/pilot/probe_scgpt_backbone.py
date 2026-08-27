"""Is the scGPT backbone we load actually carrying useful features?

The repair-pilot scGPT arms overfit hard (train loss -> 0.02, external accuracy
~0.3) and freezing the bottom 6 layers made them WORSE, not better. Those two
facts together are the signature of a backbone whose weights load by name and
shape but do not mean what the architecture thinks they mean -- the prime
suspect being the flash-attn `self_attn.Wqkv` -> `self_attn.in_proj_*` rename in
pilot_finetune_scgpt.py, which is shape-compatible and therefore silent.

This decides it without training anything:

  * embed the eval cells with the PRETRAINED backbone, no fine-tuning at all
  * classify those embeddings with logistic regression + kNN (5-fold)

If the frozen backbone lands well above chance, the features are real and the
problem is purely the fine-tuning recipe. If it lands near chance, the load is
wrong and no recipe will save it.

The `--scramble-qkv` control loads the SAME checkpoint with Q/K/V deliberately
permuted. If scrambling changes nothing, the attention weights were never
being used meaningfully in the first place.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from pilot_finetune_scgpt import (FILES, PAD_TOKEN, PAD_VALUE, S, DEFAULT_HVG,
                                  ens2sym_map, load_pilot, preprocess, tokenize)


def build(vocab, margs, n_cls):
    return TransformerModel(
        ntoken=len(vocab), d_model=margs.get("embsize", 512),
        nhead=margs.get("nheads", 8), d_hid=margs.get("d_hid", 512),
        nlayers=margs.get("nlayers", 12), nlayers_cls=3, n_cls=n_cls,
        vocab=vocab, dropout=0.0, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
        do_mvc=False, do_dab=False, use_batch_labels=False,
        domain_spec_batchnorm=False, input_emb_style="continuous",
        n_input_bins=margs.get("n_bins", 51) + 2, cell_emb_style="cls",
        use_fast_transformer=False)


def remap(sd, scramble=False, d_model=512):
    """The rename used by the training script, plus an optional Q/K/V scramble."""
    out = {}
    for k, v in sd.items():
        nk = (k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
               .replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias"))
        if scramble and "in_proj_weight" in nk:
            q, kk, vv = v[:d_model], v[d_model:2 * d_model], v[2 * d_model:]
            v = torch.cat([vv, q, kk], dim=0)      # deliberately wrong order
        out[nk] = v
    return out


def embed(model, tok, batch, device, pad_id):
    """CLS-token cell embedding straight out of the encoder (cell_emb_style='cls').

    pad_id is passed in, never read off the model: TransformerModel has no `.vocab`,
    which is exactly how the eval-time mask bug got in.
    """
    model.eval()
    ds = TensorDataset(tok["genes"], tok["values"])
    outs = []
    with torch.no_grad():
        for g, v in DataLoader(ds, batch_size=batch):
            g, v = g.to(device), v.to(device).float()
            h = model._encode(g, v, src_key_padding_mask=g.eq(pad_id))
            outs.append(h[:, 0, :].float().cpu())
    return torch.cat(outs).numpy()


def score(X, y, tag):
    # drop classes too small to stratify; they cannot inform a 5-fold estimate
    vals, cnt = np.unique(y, return_counts=True)
    keep = np.isin(y, vals[cnt >= 5])
    X, y = X[keep], y[keep]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = {"logreg": [], "knn": []}
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(max_iter=2000, n_jobs=-1).fit(X[tr], y[tr])
        accs["logreg"].append(lr.score(X[te], y[te]))
        kn = KNeighborsClassifier(n_neighbors=15).fit(X[tr], y[tr])
        accs["knn"].append(kn.score(X[te], y[te]))
    maj = np.bincount(y).max() / len(y)
    print(f"  {tag:16s} logreg {np.mean(accs['logreg']):.3f}  "
          f"knn {np.mean(accs['knn']):.3f}  (majority-class {maj:.3f}, "
          f"{len(np.unique(y))} classes, n={len(y):,})", flush=True)
    return float(np.mean(accs["logreg"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="aida", choices=list(FILES))
    ap.add_argument("--model-dir",
                    default=str(pathlib.Path.home() /
                                "data/fperalta/fresh_scGPT/scGPT_human"))
    ap.add_argument("--n-hvg", type=int, default=DEFAULT_HVG)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=6000)
    args = ap.parse_args()
    c, device = args.cohort, ("cuda" if torch.cuda.is_available() else "cpu")
    mdir = pathlib.Path(args.model_dir)

    vocab = GeneVocab.from_file(mdir / "vocab.json")
    for t in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)
    vocab.set_default_index(vocab[PAD_TOKEN])
    e2s = ens2sym_map()
    labels = json.loads((S / f"{c}_labels.json").read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    hvg_src = preprocess(load_pilot(FILES[c]["prop"], "prop", vocab, e2s),
                         n_hvg=args.n_hvg)
    genes = hvg_src.var_names.tolist()
    del hvg_src
    a = preprocess(load_pilot(FILES[c]["eval"], "eval", vocab, e2s),
                   gene_subset=genes)
    keep = a.obs["cell_type"].astype(str).isin(label2id).to_numpy()
    a._inplace_subset_obs(keep)
    if a.n_obs > args.max_cells:      # probing is O(cells); a subsample decides it
        idx = np.random.default_rng(0).choice(a.n_obs, args.max_cells, replace=False)
        a._inplace_subset_obs(np.sort(idx))
    y = np.array([label2id[t] for t in a.obs["cell_type"].astype(str)])
    tok = tokenize(a, vocab, len(genes) + 1)
    print(f"{c}: probing {a.n_obs:,} cells, {len(set(y))} classes", flush=True)

    margs = json.loads((mdir / "args.json").read_text())
    sd = torch.load(mdir / "best_model.pt", map_location="cpu")
    nw = sum(1 for k in sd if "Wqkv" in k)
    print(f"checkpoint: {len(sd)} tensors, {nw} Wqkv keys "
          f"({'flash-attn layout, rename REQUIRED' if nw else 'already in_proj'})",
          flush=True)

    pad_id = vocab[PAD_TOKEN]
    print(f"pad '{PAD_TOKEN}' = id {pad_id}; id 0 = gene "
          f"'{vocab.lookup_token(0)}'; pad covers "
          f"{tok['genes'].eq(pad_id).float().mean():.1%} of positions", flush=True)

    # (label, scramble-qkv | None to skip loading, pad id used for the mask)
    for tag, scr, pid in [("pretrained", False, pad_id),
                          ("pretrained WRONG-mask", False, 0),
                          ("qkv-scrambled", True, pad_id),
                          ("random-init", None, pad_id)]:
        model = build(vocab, margs, len(labels))
        if scr is not None:
            msd = model.state_dict()
            src = remap(sd, scramble=scr, d_model=margs.get("embsize", 512))
            loadable = {k: v for k, v in src.items()
                        if k in msd and v.shape == msd[k].shape}
            msd.update(loadable)
            model.load_state_dict(msd)
        model.to(device)
        X = embed(model, tok, args.batch, device, pid)
        score(X, y, tag)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
