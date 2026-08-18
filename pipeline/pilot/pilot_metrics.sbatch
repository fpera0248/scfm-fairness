#!/bin/bash
#SBATCH -J pilot_metrics
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_metrics_%j.out
# Integration metrics (ethnicity iLISI + cell-type ASW) for all 3 cohorts,
# arms: pretrained / P / B / P2BA on the shared external-validation cells.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
MODEL=${GF_MODEL_DIR:-$HOME/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M}
RUNPY="conda run --no-capture-output -n geneformer310 python"
for C in crc aida ild; do
  $RUNPY "$PIPE/pilot_embed_metrics.py" --cohort $C --model-dir "$MODEL"
done
echo "METRICS COMPLETE"
