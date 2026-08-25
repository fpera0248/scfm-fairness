#!/bin/bash
#SBATCH -J pilot_scgpt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_scgpt_%j.out
# scGPT repair pilot: P -> B -> P2BA for $COHORT (default aida), mirroring the
# Geneformer arms. Same data, labels, and metrics.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt310/lib/libstdc++.so.6
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
C=${COHORT:-aida}
R=/oscar/scratch/fperalta/pilot_repair/runs_scgpt/$C
RUNPY="python"

arm() { # name cond [init]
  local o=$R/$1
  if [ -f "$o/final_eval_pergroup.csv" ]; then echo "skip $1 (done)"; return; fi
  $RUNPY "$PIPE/pilot_finetune_scgpt.py" --cohort $C --cond $2 \
    ${3:+--init-from "$3"} --outdir "$o" --epochs 16 --batch 16 --freeze-layers 0 --seed 0
}

echo "=== scGPT cohort $C ==="
arm P prop
arm B ba
arm P2BA ba "$R/P"
echo "SCGPT COHORT $C COMPLETE"
