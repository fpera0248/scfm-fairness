#!/bin/bash
#SBATCH -J dump_preds
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/dump_preds_%j.out
# Backfill per-cell predictions for arms that finished BEFORE the eval scripts
# learned to dump them.
#
# Why this matters more than the checkpoints: aggregate tables cannot be turned
# back into donor-bootstrap CIs, confusion matrices, or per-group x per-cell-type
# breakdowns. Re-deriving those needs the checkpoint, and the checkpoints stay on
# scratch and go away with cluster access. The predictions are ~1MB per arm and
# go into git, so every one of those analyses stays possible forever.
#
# Not covered here (they emit predictions natively now):
#   scGPT            - job 5425786 runs pilot_perclass_scgpt.py at the end
#   scFoundation FT  - job 5427189 picks up the patch for arms it has not reached
set -uo pipefail
module load miniforge3/25.3.0-3 2>/dev/null || true
cd /oscar/home/fperalta/data/fperalta
PIPE=$HOME/data/fperalta/scfm-fairness/pipeline
S=/oscar/scratch/fperalta/pilot_repair
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "########## Geneformer: 3 cohorts x 6 arms ##########"
GF="conda run --no-capture-output -n geneformer310 python"
for C in crc aida ild; do
  for A in P B P2BA P2BJ P2BU P2DS; do
    O=$S/runs/$C/$A
    [ -d "$O" ] || { echo "skip $C $A (no run dir)"; continue; }
    if [ -f "$O/final_eval_predictions.csv" ]; then
      echo "skip $C $A (predictions exist)"; continue
    fi
    echo "=== geneformer $C $A ==="
    # --bootstrap 10, NOT 200. donor_bootstrap runs n_boot x f1_score over ~12k
    # rows per group in pure Python, which cost ~1h per arm and timed this job
    # out at 12/18 arms. The CIs it produces are now redundant anyway: the dumped
    # predictions get bootstrapped offline by analyze_predictions.py, paired and
    # with 2000 resamples. Here we only need the predictions file.
    $GF "$PIPE/eval_pergroup.py" final --run-dir "$O" \
      --dataset "$S/${C}_eval.dataset" --bootstrap 10 --batch 64 \
      || echo "  $C $A FAILED, continuing"
  done
done

echo "########## scFoundation frozen: re-run (embeddings cached, cheap) ##########"
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate scfoundation_gpu
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/users/fperalta/.conda/envs/scfoundation_gpu/lib/libstdc++.so.6
export HF_HOME=/oscar/scratch/fperalta/hf_cache
export PYTHONPATH="$PIPE/pilot:${PYTHONPATH:-}"
for C in crc aida ild; do
  if [ -f "$S/runs_scfoundation_aligned/$C/final_eval_predictions.csv" ]; then
    echo "skip scfa $C (predictions exist)"; continue
  fi
  echo "=== scfoundation-frozen $C ==="
  python "$PIPE/pilot/pilot_scfoundation_aligned.py" --cohort $C \
    || echo "  $C FAILED, continuing"
done

echo "########## inventory ##########"
find "$S" -name final_eval_predictions.csv -printf '%10s %p\n' | sort -k2
echo "DUMP PREDICTIONS COMPLETE"
