"""Fine-tune Geneformer on ONE pilot condition, optionally initialized from a
prior fine-tuned checkpoint (--init-from) for the stage-2 "repair" arms.

Deliberate differences from pipeline/finetune_geneformer.py:
- FIXED per-cohort label space (labels.json): stage-1 and stage-2 heads must
  share one geometry, so there is NO class-coverage drop here — a class with
  no train support simply stays unlearned, which is part of what we measure.
- The donor-held validation split draws from REAL-cell donors only; synthetic
  cells never enter the selection split (their labels are generator output).
Selection stays post-hoc (eval_pergroup.py select), LaBonte worst-class rule.
"""
import argparse
import collections
import json
import pathlib
import pickle

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

try:
    from geneformer import DataCollatorForCellClassification as Collator
except ImportError as e:
    raise SystemExit(f"geneformer package required for the collator: {e}")

DEFAULT_TOKEN_DICT = str(pathlib.Path.home() / "data/fperalta/Geneformer/"
                         "geneformer_repo/geneformer/token_dictionary_gc104M.pkl")


def make_collator(token_dict_path):
    try:
        with open(token_dict_path, "rb") as f:
            td = pickle.load(f)
        return Collator(token_dictionary=td)
    except TypeError:
        return Collator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", required=True)
    ap.add_argument("--labels-json", required=True)
    ap.add_argument("--coarse-map", required=True,
                    help="identity map csv; kept for eval_pergroup's contract")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--init-from", default=None,
                    help="stage-1 checkpoint dir; if set, weights start there "
                         "instead of the pretrained model")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--freeze-layers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-donor-frac", type=float, default=0.15)
    ap.add_argument("--token-dict", default=DEFAULT_TOKEN_DICT)
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ds = load_from_disk(args.tokenized)
    labels = json.loads(pathlib.Path(args.labels_json).read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    n0 = len(ds)
    ds = ds.filter(lambda b: [c in label2id for c in b["cell_type"]],
                   batched=True, num_proc=4)
    if len(ds) < n0:
        print(f"dropped {n0 - len(ds):,} cells outside the fixed label space",
              flush=True)
    ds = ds.map(lambda b: {"label": [label2id[c] for c in b["cell_type"]]},
                batched=True, num_proc=4)

    # donor-held validation from REAL donors only
    real_donors = sorted({d for d, s in zip(ds["donor_key"], ds["source"])
                          if s == "real"})
    if len(real_donors) >= 4:
        n_val = max(1, int(len(real_donors) * args.val_donor_frac))
        val_donors = set(rng.choice(real_donors, size=n_val, replace=False))
        val = ds.filter(lambda b: [d in val_donors and s == "real"
                                   for d, s in zip(b["donor_key"], b["source"])],
                        batched=True, num_proc=4)
        train = ds.filter(lambda b: [d not in val_donors
                                     for d in b["donor_key"]],
                          batched=True, num_proc=4)
    else:  # tiny downsampled pilots may have too few donors to split on
        print(f"only {len(real_donors)} real donors -- falling back to a "
              f"10% random-cell validation split", flush=True)
        val_donors = set()
        split = ds.train_test_split(test_size=0.10, seed=args.seed)
        train, val = split["train"], split["test"]

    counts = collections.Counter(train["label"])
    missing = [labels[i] for i in range(len(labels)) if counts.get(i, 0) == 0]
    if missing:
        print(f"NOTE: {len(missing)} classes have no train support here and "
              f"stay unlearned (fixed head): {missing}", flush=True)

    (out / "run_config.json").write_text(json.dumps({
        **vars(args),
        "coarse_map": str(pathlib.Path(args.coarse_map).resolve()),
        "n_train": len(train), "n_val": len(val),
        "n_labels": len(labels), "labels": labels,
        "n_val_donors": len(val_donors)}, indent=2))
    print(f"train={len(train):,} val={len(val):,} labels={len(labels)} "
          f"val_donors={len(val_donors)} init={args.init_from or 'pretrained'}",
          flush=True)

    src = args.init_from or args.model_dir
    model = BertForSequenceClassification.from_pretrained(
        src, num_labels=len(labels), ignore_mismatched_sizes=True)
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

    val.save_to_disk(str(out / "val_with_meta"))
    keep = ["input_ids", "length", "label"]
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    val_t = val.remove_columns([c for c in val.column_names if c not in keep])
    train = train.flatten_indices(num_proc=4)
    val_t = val_t.flatten_indices(num_proc=4)

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        # fp32 predict path: 64 is the proven eval ceiling on these GPUs
        per_device_eval_batch_size=min(64, args.batch * 2),
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.epochs,
        logging_steps=20,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=False,
        dataloader_num_workers=2,
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
