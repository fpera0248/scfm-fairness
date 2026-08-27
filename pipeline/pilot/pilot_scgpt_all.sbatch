#!/bin/bash
#SBATCH -J scgpt_all
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/scgpt_all_%j.out
# Retrain every scGPT repair arm after the eval-time padding-mask fix.
#
# The old runs under runs_scgpt/ are NOT reusable. predict() masked token id 0
# (the gene A1BG) instead of <pad> (id 60694), so validation, checkpoint
# selection, and final eval all ran through a corrupted mask while training
# masked correctly. best_model.pt was therefore chosen by a meaningless score.
# New results land in runs_scgpt_fixed/ so the broken runs stay on disk for the
# before/after comparison.
#
# Covers all three cohorts and all six arms, including P2BJ, which is the arm
# that actually moves rare cell types under Geneformer.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt310/lib/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
S=/oscar/scratch/fperalta/pilot_repair
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"

python -c 'import scgpt, torch; print("ENV_OK", torch.__version__, torch.cuda.is_available())'

for C in aida crc ild; do
  # P2BJ needs the joint-balanced h5ad; Geneformer's tokenized copy is not readable
  # here. Deterministic (seed 0), so this reproduces the exact set Geneformer used.
  [ -f "$S/${C}_bj.h5ad" ] || python "$PIPE/pilot_joint_balance.py" --cohort $C --h5ad-only

  R=$S/runs_scgpt_fixed/$C
  arm() { # name cond [init]
    local o=$R/$1
    if [ -f "$o/final_eval_pergroup.csv" ]; then echo "skip $C $1 (done)"; return; fi
    echo "=== $C $1 (cond=$2) ==="
    python "$PIPE/pilot_finetune_scgpt.py" --cohort $C --cond $2 \
      ${3:+--init-from "$3"} --outdir "$o" \
      --epochs 16 --batch 32 --freeze-layers 0 --seed 0 \
      || { echo "  $C $1 FAILED, continuing"; return; }
  }
  echo "########## scGPT cohort $C ##########"
  arm P    prop
  arm B    ba
  arm P2BA ba   "$R/P"
  arm P2BJ bj   "$R/P"
  arm P2BU bu   "$R/P"
  arm P2DS ds   "$R/P"
done

echo "=== per-cell-type eval on the fixed runs ==="
for C in aida crc ild; do
  python "$PIPE/pilot_perclass_scgpt.py" --cohort $C \
    --runs-root "$S/runs_scgpt_fixed" \
    --out "$S/perclass_scgpt_fixed_$C.csv" || echo "  $C perclass FAILED, continuing"
done
echo "SCGPT ALL COMPLETE"
