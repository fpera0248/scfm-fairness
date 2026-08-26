#!/bin/bash
#SBATCH -J pilot_scfa
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_scfa_%j.out
# scFoundation, GENE-ALIGNED rerun. Supersedes pilot_scfoundation.sbatch, whose
# embeddings were built on a per-cell expression-sorted permutation and so
# carried no usable gene identity.
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
for C in crc aida ild; do
  echo "=== scFoundation-aligned $C ==="
  python "$PIPE/pilot_scfoundation_aligned.py" --cohort $C || echo "  $C FAILED, continuing"
done
echo "SCFOUNDATION-ALIGNED ALL COHORTS COMPLETE"
