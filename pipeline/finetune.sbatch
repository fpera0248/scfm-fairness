#!/bin/bash
# Usage: sbatch --export=ALL,ARM=balanced,SEED=0 finetune.sbatch
#SBATCH -J bibm_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH -o bibm_ft_%j.out
#SBATCH -e bibm_ft_%j.err
set -euo pipefail
module load miniforge3/25.3.0-3
export PYTHONNOUSERSITE=1
ARM=${ARM:?set ARM=balanced|matched|proportional}
SEED=${SEED:-0}
cd /oscar/home/fperalta/data/fperalta
conda run --no-capture-output -n geneformer310 python finetune_geneformer.py \
  --arm "$ARM" --seed "$SEED" \
  --tokenized /oscar/scratch/fperalta/bibm_corpus/tokenized_v2 \
  --coarse-map celltype_coarse_map.csv \
  --model-dir ~/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M \
  --outdir /oscar/scratch/fperalta/bibm_ft/${ARM}_seed${SEED}
conda run --no-capture-output -n geneformer310 python eval_pergroup.py select \
  --run-dir /oscar/scratch/fperalta/bibm_ft/${ARM}_seed${SEED}
