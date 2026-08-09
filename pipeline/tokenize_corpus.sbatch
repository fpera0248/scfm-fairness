#!/bin/bash
#SBATCH -J bibm_tok
#SBATCH -p batch
#SBATCH -n 16
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o bibm_tok_%j.out
#SBATCH -e bibm_tok_%j.err
set -euo pipefail
module load miniforge3/25.3.0-3
export PYTHONNOUSERSITE=1
cd /oscar/scratch/fperalta/bibm_corpus
conda run --no-capture-output -n geneformer310 python /oscar/home/fperalta/data/fperalta/tokenize_corpus.py
