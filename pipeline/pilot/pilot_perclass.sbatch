#!/bin/bash
#SBATCH -J pilot_perclass
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_perclass_%j.out
# Per-cell-type accuracy/F1 on the shared eval set, all 5 arms x 3 cohorts.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
RUNPY="conda run --no-capture-output -n geneformer310 python"
for C in crc aida ild; do
  $RUNPY "$PIPE/pilot_perclass.py" --cohort $C
done
echo "PERCLASS COMPLETE"
