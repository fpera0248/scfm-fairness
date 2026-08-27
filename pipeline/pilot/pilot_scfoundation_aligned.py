"""scFoundation repair pilot, GENE-ALIGNED (supersedes pilot_scfoundation.py).

Why this file exists
--------------------
The first version reused the ASI step2a `pad_batch` convention: take the top
SEQ_LEN expressed values per cell and sort them descending. That is wrong for
scFoundation. In modelgenerator's backbone the gene identity is read off the
COLUMN POSITION, not from any token id:

    x = input_ids                                    # (B, num_genes+2) VALUES
    value_labels = x > 0
    x, x_padding = self.gatherData(x, value_labels, pad_id)
    data_gene_ids = torch.arange(self.num_genes + 2).repeat(x.shape[0], 1)
    position_gene_ids, _ = self.gatherData(data_gene_ids, value_labels, pad_id)
    x = self.token_emb(x.unsqueeze(2).float())       # value embedding
    x += self.pos_emb(position_gene_ids)             # GENE embedding <- position

So column i must be gene i of scFoundation's fixed 19,264-gene panel. Sorting by
expression permutes every cell differently, which hands the model a different
random gene assignment per cell -- the cause of the near-collinear embeddings
(pairwise corr 0.97-0.99) and the 8-11 of 32 cell-type prediction ceiling.

The fix uses modelgenerator's own alignment utilities, so the panel and the
Ensembl mapping come from the package rather than being re-derived here.

Arms and metrics are unchanged from pilot_scfoundation.py so the numbers stay
comparable to the Geneformer and scGPT pilots.
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

import pickle

S = pathlib.Path("/oscar/scratch/fperalta/pilot_repair")
ROOT = pathlib.Path.home() / "data/fperalta/scfoundation"
EMB_DIM = 768

# modelgenerator's own load_backbone_gene_list() routes symbol->Ensembl through
# bionty, which fetches an ontology from S3 and dies in this env (no s3fs, and
# compute nodes have no outbound network). The panel TSV and Geneformer's
# name->id pickle are both already on disk, so map locally and stay offline.
_MG = pathlib.Path(
    "/users/fperalta/.conda/envs/scfoundation_gpu/lib/python3.10/site-packages/"
    "modelgenerator")
PANEL_TSV = _MG / "cell/gene_lists/OS_scRNA_gene_index.19264.tsv"
NAME_ID_PKL = _MG / "huggingface_models/geneformer/gene_name_id_dict_gc95M.pkl"

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
# arm P2BJ (ancestry x cell-type joint balance), built by pilot_joint_balance.py.
# Under Geneformer this is the arm that moves rare cell types, so it has to exist
# here too or the three models cannot be compared on the arm that matters.
for _c in FILES:
    FILES[_c]["bj"] = S / f"{_c}_bj.h5ad"


def load_panel():
    """scFoundation's 19,264 gene symbols, in the order the model indexes them."""
    panel = pd.read_csv(PANEL_TSV, sep="\t")["gene_name"].astype(str).to_numpy()
    name2id = pickle.load(open(NAME_ID_PKL, "rb"))
    id2name = {v: k for k, v in name2id.items()}
    return panel, id2name


def build_projector(var_names, panel, id2name):
    """Sparse (n_our_genes x 19264) matrix placing each gene in its model column.

    Genes we can't map, and panel genes we don't carry, simply stay zero -- the
    backbone masks on `x > 0`, so absent genes cost nothing but coverage.
    """
    sym = np.array([id2name.get(str(g), "") for g in var_names])
    pos = {s: i for i, s in enumerate(panel)}
    tgt = np.array([pos.get(s, -1) for s in sym])
    keep = tgt >= 0
    # one source gene per model column, first occurrence wins (deterministic)
    seen, rows, cols = set(), [], []
    for src_i in np.flatnonzero(keep):
        t = tgt[src_i]
        if t in seen:
            continue
        seen.add(t)
        rows.append(src_i)
        cols.append(t)
    P = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(var_names), len(panel)))
    print(f"  gene alignment: {len(rows):,}/{len(panel):,} panel genes covered "
          f"({100 * len(rows) / len(panel):.1f}%)", flush=True)
    return P


def get_backbone(task):
    for attr in ("backbone", "model"):
        b = getattr(task, attr, None)
        if b is not None and hasattr(b, "_preprocess_input"):
            return b
    raise SystemExit("could not locate scFoundation backbone on the Embed task")


def embed_file(backbone, device, path, tag, cohort, panel, id2name):
    cache = S / f"scfa_{cohort}_{tag}.npz"   # scfa = scFoundation ALIGNED
    a = ad.read_h5ad(path)
    src = (a.obs["source"].astype(str).to_numpy()
           if "source" in a.obs.columns else np.full(a.n_obs, "real"))
    grp = a.obs["self_reported_ethnicity"].astype(str).str.strip().str.lower().to_numpy()
    grp = np.where(src == "synthetic", "synthetic", grp)
    ct = a.obs["cell_type"].astype(str).to_numpy()
    if tag == "eval":
        keep = ~np.isin(grp, ["synthetic", "unknown", "nan", ""])
        a, grp, ct, src = a[keep].copy(), grp[keep], ct[keep], src[keep]
    if cache.exists():
        E = np.load(cache)["E"]
        print(f"  {tag}: cached aligned embeddings {E.shape}", flush=True)
        return E, grp, ct, src

    P = build_projector(a.var_names.to_numpy(), panel, id2name)
    X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
    X = (X.astype(np.float32) @ P).tocsr()   # -> (n_cells, 19264) in model order
    n = a.n_obs
    free = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0
    bs = 32 if free > 20 else 16 if free > 10 else 8
    E = np.zeros((n, EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            x = torch.tensor(X[s:e].toarray(), dtype=torch.float32, device=device)
            x = backbone._preprocess_input(x)   # appends the 2 total-count tokens
            with torch.cuda.amp.autocast(enabled=(device == "cuda"),
                                         dtype=torch.float16):
                out = backbone(x)
                E[s:e] = out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            del x, out
            if device == "cuda" and (s // bs) % 50 == 0:
                torch.cuda.empty_cache()
    np.savez_compressed(cache, E=E)
    print(f"  {tag}: embedded {E.shape} -> {cache.name}", flush=True)
    return E, grp, ct, src


def collapse_report(E, tag):
    """The diagnostic that caught the bug: how distinct are the cell embeddings?"""
    idx = np.random.default_rng(0).choice(len(E), size=min(400, len(E)), replace=False)
    C = np.corrcoef(E[idx])
    off = C[~np.eye(len(C), dtype=bool)]
    print(f"  [{tag}] mean pairwise corr {off.mean():.3f} "
          f"(bad run was 0.966-0.991)", flush=True)


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
    for _ in range(epochs):
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
    task = Embed.from_config({"model.backbone": "scfoundation"}).to(device).eval()
    backbone = get_backbone(task)
    panel, id2name = load_panel()
    print(f"scFoundation loaded; reference panel = {len(panel)} genes", flush=True)

    labels = json.loads((S / f"{c}_labels.json").read_text())
    l2i = {l: i for i, l in enumerate(labels)}

    data = {}
    tags = ["prop", "ba", "bu", "ds", "bj", "eval"]
    if not FILES[c]["bj"].exists():
        # say so rather than quietly producing a 5-arm table that looks complete
        print(f"WARNING: {FILES[c]['bj']} missing; skipping arm P2BJ. Build it "
              f"with pilot_joint_balance.py --cohort {c} --h5ad-only", flush=True)
        tags.remove("bj")
    for tag in tags:
        E, grp, ct, src = embed_file(backbone, device, FILES[c][tag], tag, c,
                                     panel, id2name)
        collapse_report(E, tag)
        keep = np.array([t in l2i for t in ct])
        data[tag] = {"E": E[keep], "g": grp[keep], "ct": ct[keep],
                     "y": np.array([l2i[t] for t in ct[keep]])}
    del task, backbone
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
                     "n_types_predicted": int(len(set(p))),
                     **{f"f1_{g}": v for g, v in per.items()}})
        for l in sorted(set(ev["y"])):
            m = ev["y"] == l
            per_class.append({"arm": arm, "cell_type": labels[l],
                              "n_cells": int(m.sum()),
                              "accuracy": float((p[m] == l).mean())})
        print(f"{arm}: worst group {w} = {wf1:.4f}; "
              f"{len(set(p))}/{len(labels)} cell types ever predicted", flush=True)

    hP = train_head(data["prop"]["E"], data["prop"]["y"], len(labels), device,
                    epochs=args.epochs, seed=args.seed)
    evaluate(hP, "P")
    base = {k: v.clone() for k, v in hP.state_dict().items()}

    evaluate(train_head(data["ba"]["E"], data["ba"]["y"], len(labels), device,
                        epochs=args.epochs, seed=args.seed), "B")
    stage2 = [("P2BA", "ba"), ("P2BJ", "bj"), ("P2BU", "bu"), ("P2DS", "ds")]
    for arm, tag in [(a, t) for a, t in stage2 if t in data]:
        evaluate(train_head(data[tag]["E"], data[tag]["y"], len(labels), device,
                            init=base, epochs=args.epochs, seed=args.seed), arm)

    out = S / "runs_scfoundation_aligned" / c
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "final_eval_pergroup.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "final_eval_perclass.csv", index=False)
    print(f"WROTE {out}", flush=True)
    print("SCFOUNDATION-ALIGNED COHORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
