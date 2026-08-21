#!/bin/bash
#SBATCH -J pilot_probe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_probe_%j.out
# Cell-type linear probe (catastrophic-forgetting test) + Kang decoupling test,
# for pretrained / P / B / P2BA across the 3 cohorts.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
MODEL=${GF_MODEL_DIR:-$HOME/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M}
RUNPY="conda run --no-capture-output -n geneformer310 python"
for C in crc aida ild; do
  $RUNPY "$PIPE/pilot_probe.py" --cohort $C --model-dir "$MODEL"
done
echo "PROBE COMPLETE"
