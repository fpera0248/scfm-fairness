"""Third scFM: scFoundation repair pilot (all arms, one cohort).

scFoundation is reached through modelgenerator's Lightning `Embed` task (not a
plain nn.Module), so the faithful + cheap design is a FROZEN backbone with a
trainable head per arm — which is also exactly the head-vs-representation setup
our DFR analysis uses. Embeddings are computed ONCE per pilot file and cached.

Arms (identical semantics to Geneformer/scGPT):
  P     head trained on the imbalanced (prop) set
  B     head trained on the balanced-augmented set
  P2BA  P's head, further trained on balanced-augmented   (repair)
  P2BU  P's head, further trained on balanced-upsampled
  P2DS  P's head, further trained on balanced-downsampled

Embedding conventions copied verbatim from the ASI step2a script:
SEQ_LEN 15000, top-expressed genes sorted desc, fp16 autocast, mean-pooled
last_hidden_state (768-d).
"""
import argparse
import json
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
ROOT = pathlib.Path.home() / "data/fperalta/scfoundation"
SEQ_LEN, EMB_DIM = 15000, 768

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


def pad_batch(dense):  # verbatim from ASI step2a
    n_cells, n_genes = dense.shape
    take = min(n_genes, SEQ_LEN)
    if n_genes > SEQ_LEN:
        idx = np.argpartition(dense, -take, axis=1)[:, -take:]
        vals = np.take_along_axis(dense, idx, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]
        top = np.take_along_axis(vals, order, axis=1)
    else:
        order = np.argsort(dense, axis=1)[:, ::-1]
        top = np.take_along_axis(dense, order, axis=1)
    out = np.zeros((n_cells, SEQ_LEN), dtype=np.int32)
    out[:, :take] = top.astype(np.int32)
    zr = out[:, 0] == 0
    out[zr, 0] = 1
    return out


def embed_file(model, device, path, tag, cohort):
    cache = S / f"scf_{cohort}_{tag}.npz"
    a = ad.read_h5ad(path)
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    grp = a.obs["self_reported_ethnicity"].astype(str).str.strip().str.lower().to_numpy()
    grp = np.where(src == "synthetic", "synthetic", grp)
    dk = np.where(src == "synthetic", "synthetic", a.obs["donor_id"].astype(str))
    ct = a.obs["cell_type"].astype(str).to_numpy()
    if tag == "eval":
        keep = ~np.isin(grp, ["synthetic", "unknown", "nan", ""])
        a, grp, dk, ct, src = a[keep].copy(), grp[keep], dk[keep], ct[keep], src[keep]
    if cache.exists():
        E = np.load(cache)["E"]
        print(f"  {tag}: cached embeddings {E.shape}", flush=True)
        return E, grp, dk, ct, src

    X = a.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    free = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0
    bs = 64 if free > 20 else 32 if free > 10 else 16
    E = np.zeros((a.n_obs, EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, a.n_obs, bs):
            e = min(s + bs, a.n_obs)
            ids = torch.tensor(pad_batch(X[s:e].toarray()), device=device,
                               dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(device == "cuda"),
                                         dtype=torch.float16):
                out = model({"input_ids": ids})
                E[s:e] = out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            del ids, out
            if device == "cuda" and (s // bs) % 50 == 0:
                torch.cuda.empty_cache()
    np.savez_compressed(cache, E=E)
    print(f"  {tag}: embedded {E.shape} -> {cache.name}", flush=True)
    return E, grp, dk, ct, src


class Head(torch.nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(EMB_DIM), torch.nn.Linear(EMB_DIM, 256),
            torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, n_cls))

    def forward(self, x):
        return self.net(x)


def train_head(E, y, n_cls, device, init=None, epochs=40, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    h = Head(n_cls).to(device)
    if init is not None:
        h.load_state_dict(init)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(E, device=device)
    yt = torch.tensor(y, device=device, dtype=torch.long)
    n = len(y)
    for ep in range(epochs):
        h.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            loss = lf(h(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return h


def worst_group(groups, y, p):
    df = pd.DataFrame({"g": groups, "y": y, "p": p})
    sets = [set(s.y.unique()) for _, s in df.groupby("g")]
    common = sorted(set.intersection(*sets)) if sets else sorted(df.y.unique())
    per = {g: f1_score(s.y, s.p, labels=common or sorted(df.y.unique()),
                       average="macro") for g, s in df.groupby("g")}
    w = min(per, key=per.get)
    return w, per[w], per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(FILES))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    c = args.cohort
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from modelgenerator.tasks import Embed
    model = Embed.from_config({"model.backbone": "scfoundation"}).to(device).eval()
    print("scFoundation backbone loaded", flush=True)

    labels = json.loads((S / f"{c}_labels.json").read_text())
    l2i = {l: i for i, l in enumerate(labels)}

    data = {}
    for tag in ["prop", "ba", "bu", "ds", "eval"]:
        E, grp, dk, ct, src = embed_file(model, device, FILES[c][tag], tag, c)
        keep = np.array([t in l2i for t in ct])
        data[tag] = {"E": E[keep], "g": grp[keep], "ct": ct[keep],
                     "y": np.array([l2i[t] for t in ct[keep]])}
    del model
    torch.cuda.empty_cache()

    ev = data["eval"]
    Xe = torch.tensor(ev["E"], device=device)
    rows, per_class = [], []

    def evaluate(h, arm):
        h.eval()
        with torch.no_grad():
            p = h(Xe).argmax(-1).cpu().numpy()
        w, wf1, per = worst_group(ev["g"], ev["y"], p)
        rows.append({"arm": arm, "worst_group": w, "worst_group_f1": wf1,
                     **{f"f1_{g}": v for g, v in per.items()}})
        for l in sorted(set(ev["y"])):
            m = ev["y"] == l
            per_class.append({"arm": arm, "cell_type": labels[l],
                              "n_cells": int(m.sum()),
                              "accuracy": float((p[m] == l).mean())})
        print(f"{arm}: worst group {w} = {wf1:.4f}", flush=True)

    hP = train_head(data["prop"]["E"], data["prop"]["y"], len(labels), device,
                    epochs=args.epochs, seed=args.seed)
    evaluate(hP, "P")
    base = {k: v.clone() for k, v in hP.state_dict().items()}

    evaluate(train_head(data["ba"]["E"], data["ba"]["y"], len(labels), device,
                        epochs=args.epochs, seed=args.seed), "B")
    for arm, tag in [("P2BA", "ba"), ("P2BU", "bu"), ("P2DS", "ds")]:
        evaluate(train_head(data[tag]["E"], data[tag]["y"], len(labels), device,
                            init=base, epochs=args.epochs, seed=args.seed), arm)

    out = S / "runs_scfoundation" / c
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "final_eval_pergroup.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "final_eval_perclass.csv", index=False)
    print(f"WROTE {out}", flush=True)
    print("SCFOUNDATION COHORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
