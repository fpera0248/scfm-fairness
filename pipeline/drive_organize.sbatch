#!/bin/bash
#SBATCH -J drive_org
#SBATCH -p batch
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o /oscar/home/fperalta/data/fperalta/drive_org_%j.out
# Put everything offloaded from Oscar under one named folder in Drive, then sweep
# up anything written after the main upload passes started (late prediction
# dumps, mostly).
#
# Deliberately dependent on BOTH the upload and dump_preds: moving the
# destination while rclone is writing to it makes rclone recreate the folder at
# the Drive root, silently splitting the data across two places.
#
# The move is server-side. rclone's Drive backend implements DirMove, so
# reparenting the folder is one API call -- no re-upload of 285GB.
set -uo pipefail
S=/oscar/scratch/fperalta/pilot_repair
DEST="gdrive:Brown oscar Offload Aug 2026"
R=~/bin/rclone

echo "=== target: $DEST"
$R mkdir "$DEST"

if $R lsd "gdrive:" 2>/dev/null | grep -q 'scfm_pilot_repair'; then
  echo "=== moving scfm_pilot_repair into the folder (server-side)"
  $R moveto gdrive:scfm_pilot_repair "$DEST/scfm_pilot_repair" -v
  echo "move rc=$?"
else
  echo "=== scfm_pilot_repair not at root; assuming already moved"
fi

echo "=== final sweep: anything written since the upload passes began"
$R copy "$S" "$DEST/scfm_pilot_repair" \
  --transfers 8 --checkers 16 --drive-chunk-size 128M \
  --stats 300s --stats-one-line --retries 10 --low-level-retries 20
echo "sweep rc=$?"

echo "=== final state"
$R lsd "gdrive:" | grep -iE 'Brown oscar|scfm_pilot' || true
$R size "$DEST/scfm_pilot_repair"
echo "--- source for comparison:"
du -sh "$S"
echo "DRIVE ORGANIZE COMPLETE"
