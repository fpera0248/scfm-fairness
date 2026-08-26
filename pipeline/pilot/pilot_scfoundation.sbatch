#!/bin/bash
#SBATCH -J pilot_scf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_scf_%j.out
# Third scFM: scFoundation, all 5 arms, all 3 cohorts (frozen backbone + heads).
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
# weights come from HF Hub via modelgenerator; keep the cache OFF the home quota
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNPY="conda run --no-capture-output -n scfoundation_gpu python"
for C in aida crc ild; do
  echo "=== scFoundation $C ==="
  $RUNPY "$PIPE/pilot_scfoundation.py" --cohort $C || echo "  $C FAILED, continuing"
done
echo "SCFOUNDATION ALL COHORTS COMPLETE"
