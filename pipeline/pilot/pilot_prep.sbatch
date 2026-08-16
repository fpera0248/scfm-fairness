#!/bin/bash
#SBATCH -J pilot_prep
#SBATCH -p batch
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/pilot_prep_%j.out
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
conda run --no-capture-output -n geneformer310 python "$PIPE/pilot_prep.py"
