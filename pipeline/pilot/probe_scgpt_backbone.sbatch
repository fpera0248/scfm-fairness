#!/bin/bash
#SBATCH -J probe_scgpt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/probe_scgpt_%j.out
# Decides whether the scGPT repair arms are bad because of the fine-tuning recipe
# or because the pretrained backbone is not loading meaningfully. No training.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scgpt310
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scgpt310/lib/libstdc++.so.6
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"
python -c 'import scgpt, torch; print("ENV_OK", torch.__version__, torch.cuda.is_available())'
python "$PIPE/probe_scgpt_backbone.py" --cohort aida
echo "PROBE COMPLETE"
