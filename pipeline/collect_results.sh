#!/bin/bash
# Copy every result TABLE off /oscar/scratch into the repo so the pilot survives
# losing Oscar access. Scratch is also purge-eligible, so this is the only copy
# that persists.
#
# Takes .csv and .json only. Deliberately NOT the artifacts that can be rebuilt
# from code plus these tables, and that would blow past GitHub's limits:
#   *.pt      arm checkpoints, ~200-400MB each, ~40 of them
#   *.npz     cached scFoundation embeddings, ~40MB each
#   *.h5ad    joint-balanced training sets (regenerate: pilot_joint_balance.py,
#             seeded, so byte-identical)
#   *.dataset Geneformer tokenized corpora
#
# Usage:  bash pipeline/collect_results.sh
set -euo pipefail

S=${S:-/oscar/scratch/fperalta/pilot_repair}
REPO=${REPO:-$HOME/data/fperalta/scfm-fairness}
DEST="$REPO/results/pilot_repair"
LOGS="$REPO/results/slurm_logs"
MAXK=4096          # per-file cap (KB); no legitimate table here is near this
MAXTOTALM=90       # bail out before we get anywhere near a GitHub push limit

[ -d "$S" ] || { echo "no such scratch dir: $S" >&2; exit 1; }
mkdir -p "$DEST" "$LOGS"

echo "=== collecting tables from $S"
n=0
cd "$S"
while IFS= read -r -d '' f; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp -p "$f" "$DEST/$f"
  n=$((n + 1))
done < <(find . \( -name '*.csv' -o -name '*.json' \) \
              -not -path '*.dataset/*' -not -path './stage_*' \
              -size -${MAXK}k -print0)
echo "  copied $n tables"

echo "=== collecting slurm logs"
m=0
for f in "$HOME"/data/fperalta/*.out; do
  [ -e "$f" ] || continue
  # skip anything unexpectedly huge; logs are provenance, not payload
  [ "$(du -k "$f" | cut -f1)" -gt "$MAXK" ] && { echo "  SKIP oversized $(basename "$f")"; continue; }
  cp -p "$f" "$LOGS/"
  m=$((m + 1))
done
echo "  copied $m logs"

total=$(du -sm "$REPO/results" | cut -f1)
echo "=== results/ is now ${total}MB"
if [ "$total" -gt "$MAXTOTALM" ]; then
  echo "ABORT: results/ exceeds ${MAXTOTALM}MB -- something large slipped in." >&2
  echo "Inspect with: du -sm $REPO/results/* | sort -rn | head" >&2
  exit 1
fi

# A manifest matters more than usual here: once Oscar access is gone, this is the
# only record of which run directory produced which numbers.
{
  echo "# Repair pilot results"
  echo
  echo "Collected $(date -u '+%Y-%m-%d %H:%M UTC') from \`$S\`."
  echo "Tables only. Checkpoints, embedding caches, and h5ads stayed on scratch;"
  echo "everything here is regenerable from the code in \`pipeline/pilot/\`."
  echo
  echo '## Per-cell-type accuracy (the cross-model comparison)'
  echo
  echo '| model | file | arms |'
  echo '|---|---|---|'
  for f in perclass_crc.csv perclass_aida.csv perclass_ild.csv; do
    [ -f "$DEST/$f" ] && echo "| Geneformer | \`$f\` | $(tail -n +2 "$DEST/$f" | cut -d, -f1 | sort -u | tr '\n' ' ') |"
  done
  for f in perclass_scgpt_fixed_aida.csv perclass_scgpt_fixed_crc.csv perclass_scgpt_fixed_ild.csv; do
    [ -f "$DEST/$f" ] && echo "| scGPT (mask-fixed) | \`$f\` | $(tail -n +2 "$DEST/$f" | cut -d, -f1 | sort -u | tr '\n' ' ') |"
  done
  for c in crc aida ild; do
    f="runs_scfoundation_aligned/$c/final_eval_perclass.csv"
    [ -f "$DEST/$f" ] && echo "| scFoundation (frozen) | \`$f\` | $(tail -n +2 "$DEST/$f" | cut -d, -f1 | sort -u | tr '\n' ' ') |"
  done
  for c in crc aida ild; do
    for a in P B P2BA P2BJ P2BU P2DS; do
      f="runs_scfa_ft/$c/$a/final_eval_perclass.csv"
      [ -f "$DEST/$f" ] && echo "| scFoundation (full FT) | \`$f\` | $a |"
    done
  done
  echo
  echo '## Provenance warnings'
  echo
  echo '- `VOID_premaskfix_perclass_scgpt_*.csv` are the scGPT numbers produced'
  echo '  BEFORE the eval padding-mask fix. They are wrong. Kept only as the'
  echo '  before half of the before/after. Do not quote them.'
  echo '- `runs_scgpt/` (as opposed to `runs_scgpt_fixed/`) is likewise pre-fix.'
  echo '- Everything is seed 0, single seed, no confidence intervals.'
  echo
  echo '## Regenerating what was left behind'
  echo
  echo '```'
  echo 'python pipeline/pilot/pilot_joint_balance.py --cohort {crc,aida,ild}   # bj h5ad + tokenized set'
  echo 'sbatch pipeline/pilot/pilot_scgpt_all.sbatch                           # scGPT arms + per-cell-type'
  echo 'sbatch pipeline/pilot/pilot_scfoundation_aligned.sbatch                # scFoundation frozen'
  echo 'sbatch pipeline/pilot/pilot_scfa_finetune.sbatch                       # scFoundation full fine-tune'
  echo '```'
} > "$REPO/results/MANIFEST.md"

echo "=== wrote $REPO/results/MANIFEST.md"
echo "Now:  cd $REPO && git add -A results && git commit && git push"
