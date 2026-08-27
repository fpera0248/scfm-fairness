#!/bin/bash
#SBATCH -J scfa_backfill
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/scfa_backfill_%j.out
# scfa_ft arms that finished BEFORE the per-cell prediction dump was added get
# their final eval redone from the saved checkpoint (--eval-only, no retraining).
# Every later arm emits predictions natively, so this scans rather than hardcodes
# which arms are missing -- it stays correct however far the job got.
#
# Depends on scfa_ft (5427189) so checkpoints are not being written underneath us.
set -uo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
S=/oscar/scratch/fperalta/pilot_repair
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scfoundation_gpu/lib/libstdc++.so.6
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"

declare -A COND=( [P]=prop [B]=ba [P2BA]=ba [P2BJ]=bj [P2BU]=bu [P2DS]=ds )
n=0
for C in aida crc ild; do
  for A in P B P2BA P2BJ P2BU P2DS; do
    O=$S/runs_scfa_ft/$C/$A
    [ -f "$O/best_model.pt" ] || continue
    [ -f "$O/final_eval_predictions.csv" ] && continue
    echo "=== backfill $C $A (cond=${COND[$A]}) ==="
    python "$PIPE/pilot_scfoundation_finetune.py" --cohort $C \
      --cond "${COND[$A]}" --outdir "$O" --eval-only \
      || echo "  $C $A FAILED, continuing"
    n=$((n+1))
  done
done
echo "backfilled $n arms"
find "$S" -name final_eval_predictions.csv | wc -l
echo "SCFA BACKFILL COMPLETE"
