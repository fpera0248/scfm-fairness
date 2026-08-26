#!/bin/bash
#SBATCH -J pilot_joint
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_joint_%j.out
# Track B: ancestry x cell-type JOINTLY balanced repair (arm P2BJ), all cohorts.
# Answers Jiaqi: does balancing the GRID (not just the ancestry margin) finally
# move rare cell types?
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
EVALPY=$HOME/data/fperalta/scfm-fairness/pipeline/eval_pergroup.py
S=/oscar/scratch/fperalta/pilot_repair
MODEL=${GF_MODEL_DIR:-$HOME/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNPY="conda run --no-capture-output -n geneformer310 python"

for C in crc aida ild; do
  echo "=== $C: build joint-balanced set ==="
  [ -d "$S/${C}_bj.dataset" ] || $RUNPY "$PIPE/pilot_joint_balance.py" --cohort $C

  O=$S/runs/$C/P2BJ
  if [ -f "$O/DONE" ]; then echo "skip $C P2BJ (done)"; continue; fi
  INIT=$(ls -d $S/runs/$C/P/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)
  echo "=== $C: train P2BJ from $INIT ==="
  $RUNPY "$PIPE/pilot_finetune.py" --tokenized "$S/${C}_bj.dataset" \
    --labels-json "$S/${C}_labels.json" --coarse-map "$S/${C}_identity_map.csv" \
    --model-dir "$MODEL" --init-from "$INIT" --outdir "$O" \
    --epochs 8 --lr 1e-5 --batch 12 --grad-accum 4 --seed 0
  touch "$O/DONE"

  $RUNPY "$EVALPY" select --run-dir "$O" --batch 64
  $RUNPY "$EVALPY" final --run-dir "$O" --dataset "$S/${C}_eval.dataset" \
    --bootstrap 200 --batch 64
done
echo "JOINT BALANCE COMPLETE"
