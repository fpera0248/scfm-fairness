"""Fine-tune Geneformer on one corpus arm.

Training only. Model selection happens AFTERWARD in eval_pergroup.py over the
per-epoch checkpoints (worst-class accuracy on the donor-held validation split,
never the average) — the LaBonte rule, kept out of the training loop on purpose.
"""
import argparse
import collections
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required for the collator: {e}")

ARMS = {"balanced": "arm_balanced", "matched": "arm_matched",
        "proportional": "arm_proportional"}
VAL_DONOR_FRAC = 0.10
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


def load_coarse_map(path):
    m = pd.read_csv(path)
    return dict(zip(m.cell_type, m.coarse_label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--tokenized", required=True)
    ap.add_argument("--coarse-map", required=True)
    ap.add_argument("--model-dir", required=True, help="e.g. .../Geneformer-V2-316M")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--freeze-layers", type=int, default=12,
                    help="bottom encoder layers to freeze (V2-316M has 24)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cells", type=int, default=0, help="0 = full arm")
    ap.add_argument("--token-dict", default=DEFAULT_TOKEN_DICT,
                    help="Geneformer token dictionary pkl for the collator")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ds = load_from_disk(args.tokenized)
    coarse = load_coarse_map(args.coarse_map)

    arm_col = ARMS[args.arm]
    ds = ds.filter(lambda b: [s == "train" and a == 1
                              for s, a in zip(b["split"], b[arm_col])],
                   batched=True, num_proc=8)

    # coarse labels; drop unmapped
    ds = ds.map(lambda b: {"coarse": [coarse.get(c, "DROP") for c in b["cell_type"]]},
                batched=True, num_proc=8)
    ds = ds.filter(lambda b: [c != "DROP" for c in b["coarse"]],
                   batched=True, num_proc=8)

    labels = sorted(set(ds["coarse"]))
    label2id = {l: i for i, l in enumerate(labels)}
    ds = ds.map(lambda b: {"label": [label2id[c] for c in b["coarse"]]},
                batched=True, num_proc=8)

    # donor-held validation split for model selection (distinct from eval_donor holdout)
    donors = sorted(set(ds["donor_key"]))
    val_donors = set(rng.choice(donors, size=max(2, int(len(donors) * VAL_DONOR_FRAC)),
                                replace=False))
    val = ds.filter(lambda b: [d in val_donors for d in b["donor_key"]],
                    batched=True, num_proc=8)
    train = ds.filter(lambda b: [d not in val_donors for d in b["donor_key"]],
                      batched=True, num_proc=8)
    if args.max_cells and len(train) > args.max_cells:
        train = train.shuffle(seed=args.seed).select(range(args.max_cells))

    # class-coverage guard: a class with no TRAIN support can never be
    # learned and pins worst-class metrics to 0 forever -> drop it loudly
    # from both splits and remap ids
    present = sorted(set(train["label"]))
    if len(present) < len(labels):
        missing = [labels[i] for i in range(len(labels)) if i not in set(present)]
        print(f"dropping {len(missing)} classes with no train support: "
              f"{missing}", flush=True)
        keepset = set(present)
        remap = {old: new for new, old in enumerate(present)}
        labels = [labels[i] for i in present]
        train = train.filter(lambda b: [l in keepset for l in b["label"]],
                             batched=True, num_proc=8)
        val = val.filter(lambda b: [l in keepset for l in b["label"]],
                         batched=True, num_proc=8)
        train = train.map(lambda b: {"label": [remap[l] for l in b["label"]]},
                          batched=True, num_proc=8)
        val = val.map(lambda b: {"label": [remap[l] for l in b["label"]]},
                      batched=True, num_proc=8)

    # the ILD exclusion happened AFTER arm sampling, so balanced/matched arms
    # carry a small per-group skew -> re-equalize train counts to the floor
    if args.arm in ("balanced", "matched"):
        counts = collections.Counter(train["group"])
        floor = min(counts.values())
        if max(counts.values()) > floor:
            print(f"re-equalizing groups to {floor:,}/group "
                  f"(pre: {dict(sorted(counts.items()))})", flush=True)
            gcol = train["group"]
            rng2 = np.random.default_rng(args.seed + 1)
            keep_idx = []
            for g in counts:
                gi = np.flatnonzero(np.asarray(gcol) == g)
                keep_idx.extend(rng2.choice(gi, size=floor, replace=False).tolist())
            train = train.select(sorted(keep_idx))

    (out / "run_config.json").write_text(json.dumps({
        **vars(args),
        "coarse_map": str(pathlib.Path(args.coarse_map).resolve()),
        "n_train": len(train), "n_val": len(val),
        "n_labels": len(labels), "labels": labels,
        "n_val_donors": len(val_donors)}, indent=2))
    print(f"arm={args.arm} train={len(train):,} val={len(val):,} "
          f"labels={len(labels)} val_donors={len(val_donors)}", flush=True)

    model = BertForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=len(labels),
        ignore_mismatched_sizes=True)
    if args.freeze_layers > 0:
        for p in model.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in model.bert.encoder.layer[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def metrics(pred):
        y = pred.label_ids
        yhat = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(y, yhat),
                "macro_f1": f1_score(y, yhat, average="macro")}

    keep = ["input_ids", "length", "label"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val_t = val.remove_columns([c for c in val.column_names if c not in keep])
    # shuffle/select leave an indices mapping: every batch then gathers
    # random rows through indirection over the full arrow set on network
    # scratch (~8.5s/step observed). Materialize contiguously.
    train = train.flatten_indices(num_proc=8)
    val_t = val_t.flatten_indices(num_proc=8)
    # in-loop Trainer eval is only a progress curve; selection happens post-hoc
    # on val_with_meta -- cap the per-epoch eval cost
    if len(val_t) > 30000:
        val_t = val_t.shuffle(seed=0).select(range(30000)).flatten_indices()
    # persist val WITH metadata for post-hoc per-group selection
    val.save_to_disk(str(out / "val_with_meta"))

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 4,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        logging_steps=200,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,
        # reentrant checkpointing + frozen bottom layers silently yields
        # grad=None for ALL checkpointed layers (verified empirically) --
        # non-reentrant keeps layers 12-23 training
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=val_t, compute_metrics=metrics,
                      data_collator=make_collator(args.token_dict))
    trainer.train()
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
