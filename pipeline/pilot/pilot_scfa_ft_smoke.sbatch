#!/bin/bash
#SBATCH -J scfa_ft_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/scfa_ft_smoke_%j.out
# Two epochs of ONE arm before committing ~12h of GPU to the full 18.
# Checks: does backprop through the backbone fit in VRAM, how long is an epoch,
# does the loss actually move, and does the block finder locate the encoder.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scfoundation_gpu/lib/libstdc++.so.6
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$PIPE/pilot_scfoundation_finetune.py" --cohort aida --cond prop \
  --outdir /oscar/scratch/fperalta/pilot_repair/runs_scfa_ft_smoke/aida/P \
  --epochs 2
echo "SMOKE COMPLETE"
