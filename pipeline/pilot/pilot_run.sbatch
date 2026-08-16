#!/bin/bash
#SBATCH -J pilot_repair
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_repair_%j.out
# Requires: --export=ALL,COHORT=ild|crc|aida
# Arms: P (imbalanced only), B (balanced only), P2BA/P2BU/P2DS (imbalanced
# stage-1 -> balanced stage-2 "repair"), each selected by the LaBonte
# worst-class rule and scored on the cohort's ASI external-validation set.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
EVALPY=$HOME/data/fperalta/scfm-fairness/pipeline/eval_pergroup.py
S=/oscar/scratch/fperalta/pilot_repair
C=${COHORT:?set COHORT=ild|crc|aida}
R=$S/runs/$C
MODEL=${GF_MODEL_DIR:-$HOME/data/fperalta/Geneformer/Geneformer-V2-316M}
RUNPY="conda run --no-capture-output -n geneformer310 python"

last_ckpt() { ls -d "$1"/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1; }

ft() { # name cond [init]
  local out=$R/$1 tok=$S/${C}_$2.dataset init=${3:-}
  if [ -f "$out/DONE" ]; then echo "skip $1 (done)"; return; fi
  $RUNPY "$PIPE/pilot_finetune.py" --tokenized "$tok" \
    --labels-json "$S/${C}_labels.json" --coarse-map "$S/${C}_identity_map.csv" \
    --model-dir "$MODEL" ${init:+--init-from "$init"} \
    --outdir "$out" --epochs 8 --lr 1e-5 --batch 24 --grad-accum 2 --seed 0
  touch "$out/DONE"
}

echo "=== cohort $C: stage-1 + references ==="
ft P prop
ft B ba
INIT=$(last_ckpt "$R/P")
echo "=== stage-2 repair arms init from $INIT ==="
ft P2BA ba "$INIT"
ft P2BU bu "$INIT"
ft P2DS ds "$INIT"

echo "=== selection + final eval on the shared external-validation set ==="
for arm in P B P2BA P2BU P2DS; do
  echo "--- $arm select"
  [ -f "$R/$arm/selected.json" ] || \
    $RUNPY "$EVALPY" select --run-dir "$R/$arm" --batch 64
  echo "--- $arm final"
  $RUNPY "$EVALPY" final --run-dir "$R/$arm" \
    --dataset "$S/${C}_eval.dataset" --bootstrap 200 --batch 64
done
echo "COHORT $C COMPLETE"
