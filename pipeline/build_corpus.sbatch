#!/bin/bash
#SBATCH -J bibm_corpus
#SBATCH -p batch
#SBATCH -n 8
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o bibm_corpus_%j.out
#SBATCH -e bibm_corpus_%j.err
set -euo pipefail
module load miniforge3/25.3.0-3
cd /oscar/scratch/fperalta/bibm_corpus
conda run --no-capture-output -n census python /oscar/home/fperalta/data/fperalta/build_corpus.py all
