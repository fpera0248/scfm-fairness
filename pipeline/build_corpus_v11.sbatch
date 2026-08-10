#!/bin/bash
# Usage: sbatch build_corpus_v11.sbatch                          (DISCO only)
#        sbatch --export=ALL,ADAPTERS="disco scea" build_corpus_v11.sbatch
#SBATCH -J v11_corpus
#SBATCH -p batch
#SBATCH -n 8
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH -o v11_corpus_%j.out
#SBATCH -e v11_corpus_%j.err
set -euo pipefail
module load miniforge3/25.3.0-3
export PYTHONNOUSERSITE=1
# datasets cache spill (census joinid cast + concat) must NOT hit $HOME quota
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export HF_DATASETS_CACHE=/oscar/scratch/fperalta/hf_cache/datasets
# absolute script path (tokenize_corpus.sbatch convention): never depends on
# what happens to be in the cwd. Deploy celltype_coarse_map.csv beside the
# script (or pass --coarse-map).
cd /oscar/home/fperalta/data/fperalta
conda run --no-capture-output -n geneformer310 python \
  /oscar/home/fperalta/data/fperalta/build_corpus_v11.py \
  --adapters "${ADAPTERS:-disco}"
