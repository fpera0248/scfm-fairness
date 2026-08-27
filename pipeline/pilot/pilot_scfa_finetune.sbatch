#!/bin/bash
#SBATCH -J scfa_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/scfa_ft_%j.out
# scFoundation FULL FINE-TUNE, all cohorts x all six arms.
#
# The frozen-backbone runs (runs_scfoundation_aligned/) adapt scFoundation by
# training a head on cached embeddings, while Geneformer and scGPT fine-tune
# end to end. That asymmetry means a cross-model claim about a cell type could
# be about the adaptation method rather than the model. This removes it.
#
# Results land in runs_scfa_ft/; the frozen runs stay where they are, so the
# frozen-vs-finetuned contrast is itself measurable.
set -euo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline/pilot
S=/oscar/scratch/fperalta/pilot_repair
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scfoundation_gpu/lib/libstdc++.so.6
export PYTHONPATH="$PIPE:${PYTHONPATH:-}"

python -c 'import torch; print("ENV_OK", torch.__version__, torch.cuda.is_available())'

for C in aida crc ild; do
  [ -f "$S/${C}_bj.h5ad" ] || python "$PIPE/pilot_joint_balance.py" --cohort $C --h5ad-only
  R=$S/runs_scfa_ft/$C
  arm() { # name cond [init]
    local o=$R/$1
    if [ -f "$o/final_eval_pergroup.csv" ]; then echo "skip $C $1 (done)"; return; fi
    echo "=== $C $1 (cond=$2) ==="
    python "$PIPE/pilot_scfoundation_finetune.py" --cohort $C --cond $2 \
      ${3:+--init-from "$3"} --outdir "$o" --epochs 8 --seed 0 \
      || { echo "  $C $1 FAILED, continuing"; return; }
  }
  echo "########## scFoundation full fine-tune: $C ##########"
  arm P    prop
  arm B    ba
  arm P2BA ba   "$R/P"
  arm P2BJ bj   "$R/P"
  arm P2BU bu   "$R/P"
  arm P2DS ds   "$R/P"
done
echo "SCFA FINETUNE ALL COMPLETE"
