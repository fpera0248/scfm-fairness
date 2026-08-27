"""scFoundation repair pilot, FULL FINE-TUNE (companion to pilot_scfoundation_aligned.py).

Why this exists
---------------
`pilot_scfoundation_aligned.py` freezes the backbone and trains a small head on
cached embeddings. Geneformer (`pilot_finetune.py`) and scGPT
(`pilot_finetune_scgpt.py`) both fine-tune the whole backbone. So any
cross-model statement built on the frozen version confounds the MODEL with the
ADAPTATION METHOD: "scFoundation fails on this cell type" could just as well be
"a linear probe on frozen features fails on this cell type".

This trains scFoundation end to end -- same arms, same donor-held-out validation,
same LaBonte worst-class checkpoint selection, same external eval and metrics as
the other two models -- so the three are adapted comparably and the comparison
means what it says.

Gene alignment, the panel projection, and `_preprocess_input` are imported from
pilot_scfoundation_aligned so there is exactly one definition of the thing that
was wrong the first time round.

Pooling note: cell embedding is `last_hidden_state.mean(dim=1)`, identical to the
frozen path. That mean runs over padded positions too. It is kept as-is on
purpose -- changing pooling AND unfreezing the backbone in one step would make
any difference impossible to attribute. Worth revisiting separately.

Memory: the frozen path runs batch 32 under no_grad. Backprop through a 12-layer
transformer over a few thousand gathered nonzeros needs far less, so the batch is
auto-sized from free VRAM and an accumulation factor keeps the EFFECTIVE batch
fixed across cohorts and GPUs.
"""
import argparse
import json
import pathlib
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score

from pilot_scfoundation_aligned import (EMB_DIM, FILES, S, Head, build_projector,
                                        get_backbone, load_panel, worst_group)


def load_arm(path, tag, panel, id2name):
    """h5ad -> (X in scFoundation panel column order, group, cell_type, donor, source)."""
    a = ad.read_h5ad(path)
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    grp = a.obs["self_reported_ethnicity"].astype(str).str.strip().str.lower().to_numpy()
    grp = np.where(src == "synthetic", "synthetic", grp)
    ct = a.obs["cell_type"].astype(str).to_numpy()
    donor = (a.obs["donor_id"].astype(str).to_numpy()
             if "donor_id" in a.obs.columns else np.full(a.n_obs, "unknown"))
    donor = np.where(src == "synthetic", "synthetic", donor)
    if tag == "eval":
        keep = ~np.isin(grp, ["synthetic", "unknown", "nan", ""])
        a, grp, ct, src, donor = (a[keep].copy(), grp[keep], ct[keep],
                                  src[keep], donor[keep])
    P = build_projector(a.var_names.to_numpy(), panel, id2name)
    X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
    X = (X.astype(np.float32) @ P).tocsr()
    del a
    return X, grp, ct, src, donor


def encoder_blocks(backbone):
    """The list of transformer blocks, for partial unfreezing.

    Located by structure rather than by a hardcoded attribute path: take the
    longest nn.ModuleList under the backbone whose children look like repeated
    blocks. Returns None if nothing convincing is found, in which case the caller
    falls back to training everything and says so.
    """
    best = None
    for name, mod in backbone.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 4:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def set_trainable(backbone, unfreeze_last):
    """unfreeze_last < 0 trains the whole backbone (matches Geneformer/scGPT)."""
    if unfreeze_last < 0:
        for p in backbone.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in backbone.parameters())
        print(f"  backbone: FULL fine-tune, {n/1e6:.1f}M params trainable",
              flush=True)
        return
    found = encoder_blocks(backbone)
    for p in backbone.parameters():
        p.requires_grad = False
    if found is None:
        print("  WARNING: could not locate encoder blocks; training everything",
              flush=True)
        for p in backbone.parameters():
            p.requires_grad = True
        return
    name, blocks = found
    for blk in blocks[-unfreeze_last:]:
        for p in blk.parameters():
            p.requires_grad = True
    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  backbone: top {unfreeze_last}/{len(blocks)} blocks of '{name}' "
          f"trainable, {n/1e6:.1f}M params", flush=True)


def pick_batch(device, requested):
    if requested:
        return requested
    if device != "cuda":
        return 2
    free = torch.cuda.mem_get_info()[0] / 1024**3
    # backprop activations dominate; these are deliberately conservative
    return 8 if free > 60 else 4 if free > 30 else 2 if free > 16 else 1


def forward_batch(backbone, head, X, idx, device):
    x = torch.tensor(X[idx].toarray(), dtype=torch.float32, device=device)
    x = backbone._preprocess_input(x)
    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
        out = backbone(x)
        emb = out.last_hidden_state.mean(dim=1)
        return head(emb.float())


@torch.no_grad()
def predict(backbone, head, X, device, batch):
    backbone.eval(); head.eval()
    preds = []
    for s in range(0, X.shape[0], batch):
        idx = np.arange(s, min(s + batch, X.shape[0]))
        preds.append(forward_batch(backbone, head, X, idx,
                                   device).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(FILES))
    ap.add_argument("--cond", required=True,
                    choices=["prop", "ba", "bu", "ds", "bj"])
    ap.add_argument("--init-from", default=None, help="stage-1 outdir")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone lr")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from free VRAM")
    ap.add_argument("--effective-batch", type=int, default=32,
                    help="held constant via gradient accumulation")
    ap.add_argument("--unfreeze-last", type=int, default=-1,
                    help="-1 = full fine-tune (matches the other two models)")
    ap.add_argument("--val-donor-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c = args.cohort
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    labels = json.loads((S / f"{c}_labels.json").read_text())
    l2i = {l: i for i, l in enumerate(labels)}

    from modelgenerator.tasks import Embed
    task = Embed.from_config({"model.backbone": "scfoundation"}).to(device)
    backbone = get_backbone(task)
    panel, id2name = load_panel()
    print(f"scFoundation loaded; panel = {len(panel)} genes", flush=True)

    Xtr, gtr, ctr, str_, dtr = load_arm(FILES[c][args.cond], args.cond, panel, id2name)
    Xev, gev, cev, _, _ = load_arm(FILES[c]["eval"], "eval", panel, id2name)
    ktr = np.array([t in l2i for t in ctr])
    kev = np.array([t in l2i for t in cev])
    Xtr, gtr, ctr, str_, dtr = Xtr[ktr], gtr[ktr], ctr[ktr], str_[ktr], dtr[ktr]
    Xev, gev, cev = Xev[kev], gev[kev], cev[kev]
    ytr = np.array([l2i[t] for t in ctr])
    yev = np.array([l2i[t] for t in cev])
    print(f"train {Xtr.shape[0]:,} cells / eval {Xev.shape[0]:,} cells", flush=True)

    # donor-held-out validation from REAL donors, same rule as the other models
    real = sorted(set(dtr[str_ == "real"]))
    if len(real) >= 4:
        vd = set(rng.choice(real, size=max(1, int(len(real) * args.val_donor_frac)),
                            replace=False))
        is_val = np.array([d in vd and s == "real" for d, s in zip(dtr, str_)])
    else:
        is_val = rng.random(len(ytr)) < 0.10
    is_trn = ~is_val
    print(f"train={is_trn.sum():,} val={is_val.sum():,} "
          f"({len(real)} real donors)", flush=True)

    head = Head(len(labels)).to(device)
    if args.init_from:
        ck = torch.load(pathlib.Path(args.init_from) / "best_model.pt",
                        map_location="cpu")
        backbone.load_state_dict(ck["backbone"], strict=False)
        head.load_state_dict(ck["head"])
        print(f"initialized from {args.init_from}", flush=True)
    set_trainable(backbone, args.unfreeze_last)

    batch = pick_batch(device, args.batch)
    accum = max(1, args.effective_batch // batch)
    print(f"batch {batch} x accum {accum} = effective {batch * accum}", flush=True)

    opt = torch.optim.AdamW(
        [{"params": [p for p in backbone.parameters() if p.requires_grad],
          "lr": args.lr},
         {"params": head.parameters(), "lr": args.head_lr}], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    lossf = torch.nn.CrossEntropyLoss()

    (out / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "batch_used": batch, "accum": accum, "n_labels": len(labels)}, indent=2))

    tr_idx = np.flatnonzero(is_trn)
    Xval, yval = Xtr[is_val], ytr[is_val]
    best = {"worst": -1.0, "f1": -1.0, "epoch": -1}
    for ep in range(1, args.epochs + 1):
        backbone.train(); head.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = rng.permutation(tr_idx)
        opt.zero_grad(set_to_none=True)
        for step, s in enumerate(range(0, len(perm), batch)):
            idx = perm[s:s + batch]
            logits = forward_batch(backbone, head, Xtr, idx, device)
            loss = lossf(logits, torch.tensor(ytr[idx], device=device,
                                              dtype=torch.long)) / accum
            scaler.scale(loss).backward()
            tot += float(loss) * accum; nb += 1
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
        yhat = predict(backbone, head, Xval, device, batch)
        per = pd.Series(yval).groupby(yval).apply(
            lambda s: (yhat[s.index] == s.values).mean())
        res = {"epoch": ep, "loss": round(tot / max(nb, 1), 4),
               "val_acc": round(float(accuracy_score(yval, yhat)), 4),
               "val_macro_f1": round(float(f1_score(yval, yhat, average="macro")), 4),
               "val_worst_class": round(float(per.min()), 4),
               "secs": round(time.time() - t0)}
        print(res, flush=True)
        if (res["val_worst_class"], res["val_macro_f1"]) > (best["worst"], best["f1"]):
            best = {"worst": res["val_worst_class"], "f1": res["val_macro_f1"],
                    "epoch": ep}
            torch.save({"backbone": backbone.state_dict(),
                        "head": head.state_dict()}, out / "best_model.pt")
    (out / "selected.json").write_text(json.dumps(best, indent=2))
    print(f"SELECTED epoch {best['epoch']}", flush=True)

    ck = torch.load(out / "best_model.pt", map_location="cpu")
    backbone.load_state_dict(ck["backbone"], strict=False)
    head.load_state_dict(ck["head"])
    backbone.to(device); head.to(device)
    p = predict(backbone, head, Xev, device, batch)
    w, wf1, per = worst_group(gev, yev, p)
    pd.DataFrame([{"group": g, "n_cells": int((gev == g).sum()),
                   "macro_f1_common": v} for g, v in per.items()]
                 ).sort_values("macro_f1_common").to_csv(
        out / "final_eval_pergroup.csv", index=False)
    pd.DataFrame([{"cell_type": labels[l], "n_cells": int((yev == l).sum()),
                   "accuracy": float((p[yev == l] == l).mean())}
                  for l in sorted(set(yev))]).to_csv(
        out / "final_eval_perclass.csv", index=False)
    print(f"worst group: {w} = {wf1:.4f}; "
          f"{len(set(p))}/{len(labels)} cell types ever predicted", flush=True)
    print("SCFOUNDATION FINETUNE ARM COMPLETE", flush=True)


if __name__ == "__main__":
    main()
