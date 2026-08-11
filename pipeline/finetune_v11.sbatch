#!/bin/bash
# v1.1 arms: expanded corpus (tokenized_v2_1, census+DISCO+SCEA, 4,317,725
# cells), 100k cells/group after rebalance (--max-cells 500000), 96h wall
# (gpu partition MaxTime is UNLIMITED; 4 epochs at ~7.3s/step ~= 85h + evals).
# Usage: sbatch --export=ALL,ARM=balanced,SEED=0 finetune_v11.sbatch
#SBATCH -J bibm_ft_v11
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40
#SBATCH -n 6
#SBATCH --mem=192G
#SBATCH -t 96:00:00
#SBATCH -o bibm_ftv11_%j.out
#SBATCH -e bibm_ftv11_%j.err
set -euo pipefail
module load miniforge3/25.3.0-3
export PYTHONNOUSERSITE=1
ARM=${ARM:?set ARM=balanced|matched|proportional}
SEED=${SEED:-0}
cd /oscar/home/fperalta/data/fperalta
conda run --no-capture-output -n geneformer310 python finetune_geneformer.py \
  --arm "$ARM" --seed "$SEED" \
  --tokenized /oscar/scratch/fperalta/bibm_corpus/tokenized_v2_1 \
  --coarse-map celltype_coarse_map.csv \
  --model-dir ~/data/fperalta/Geneformer/geneformer_repo/Geneformer-V2-316M \
  --outdir /oscar/scratch/fperalta/bibm_ft_v11/${ARM}_seed${SEED} \
  --batch 24 --grad-accum 2 --epochs 4 --max-cells 500000
conda run --no-capture-output -n geneformer310 python eval_pergroup.py select \
  --run-dir /oscar/scratch/fperalta/bibm_ft_v11/${ARM}_seed${SEED}
