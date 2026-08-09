"""Fine-tune Geneformer on one corpus arm.

Training only. Model selection happens AFTERWARD in eval_pergroup.py over the
per-epoch checkpoints (worst-class accuracy on the donor-held validation split,
never the average) — the LaBonte rule, kept out of the training loop on purpose.
"""
import argparse
import json
import pathlib

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

    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "n_train": len(train), "n_val": len(val),
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
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      eval_dataset=val_t, compute_metrics=metrics,
                      data_collator=Collator())
    trainer.train()
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
