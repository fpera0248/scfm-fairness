#!/bin/bash
#SBATCH -J pilot_pc_scgpt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_pc_scgpt_%j.out
# Per-cell-type accuracy for the scGPT arms, all cohorts. Answers Jiaqi's
# "maybe check results from other foundation models" for the rare cell types.
# Inference only: reuses the trained 1200-HVG checkpoints, no retraining.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt_gpu
export PYTHONNOUSERSITE=1
# same CXXABI fix every ASI scGPT slurm script uses: preload the ENV's libstdc++
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt_gpu/lib/libstdc++.so.6
# pilot_perclass_scgpt imports pilot_finetune_scgpt for the gene space
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"
for C in crc aida ild; do
  echo "=== scGPT per-cell-type: $C ==="
  python "$PIPE/pilot_perclass_scgpt.py" --cohort $C || echo "  $C FAILED, continuing"
done
echo "SCGPT PERCLASS COMPLETE"
