# Repair pilot results

Collected 2026-08-29 12:14 UTC from `/oscar/scratch/fperalta/pilot_repair`.
Tables only. Checkpoints, embedding caches, and h5ads stayed on scratch;
everything here is regenerable from the code in `pipeline/pilot/`.

## Per-cell-type accuracy (the cross-model comparison)

| model | file | arms |
|---|---|---|
| Geneformer | `perclass_crc.csv` | B P P2BA P2BJ P2BU P2DS  |
| Geneformer | `perclass_aida.csv` | B P P2BA P2BJ P2BU P2DS  |
| Geneformer | `perclass_ild.csv` | B P P2BA P2BJ P2BU P2DS  |
| scGPT (mask-fixed) | `perclass_scgpt_fixed_aida.csv` | B P P2BA P2BJ P2BU P2DS  |
| scGPT (mask-fixed) | `perclass_scgpt_fixed_crc.csv` | B P P2BA P2BJ P2BU P2DS  |
| scGPT (mask-fixed) | `perclass_scgpt_fixed_ild.csv` | B P P2BA P2BJ P2BU P2DS  |
| scFoundation (frozen) | `runs_scfoundation_aligned/crc/final_eval_perclass.csv` | B P P2BA P2BJ P2BU P2DS  |
| scFoundation (frozen) | `runs_scfoundation_aligned/aida/final_eval_perclass.csv` | B P P2BA P2BJ P2BU P2DS  |
| scFoundation (frozen) | `runs_scfoundation_aligned/ild/final_eval_perclass.csv` | B P P2BA P2BJ P2BU P2DS  |
| scFoundation (full FT) | `runs_scfa_ft/crc/P/final_eval_perclass.csv` | P |
| scFoundation (full FT) | `runs_scfa_ft/crc/B/final_eval_perclass.csv` | B |
| scFoundation (full FT) | `runs_scfa_ft/crc/P2BA/final_eval_perclass.csv` | P2BA |
| scFoundation (full FT) | `runs_scfa_ft/crc/P2BJ/final_eval_perclass.csv` | P2BJ |
| scFoundation (full FT) | `runs_scfa_ft/crc/P2BU/final_eval_perclass.csv` | P2BU |
| scFoundation (full FT) | `runs_scfa_ft/crc/P2DS/final_eval_perclass.csv` | P2DS |
| scFoundation (full FT) | `runs_scfa_ft/aida/P/final_eval_perclass.csv` | P |
| scFoundation (full FT) | `runs_scfa_ft/aida/B/final_eval_perclass.csv` | B |
| scFoundation (full FT) | `runs_scfa_ft/aida/P2BA/final_eval_perclass.csv` | P2BA |
| scFoundation (full FT) | `runs_scfa_ft/aida/P2BJ/final_eval_perclass.csv` | P2BJ |
| scFoundation (full FT) | `runs_scfa_ft/aida/P2BU/final_eval_perclass.csv` | P2BU |
| scFoundation (full FT) | `runs_scfa_ft/aida/P2DS/final_eval_perclass.csv` | P2DS |
| scFoundation (full FT) | `runs_scfa_ft/ild/P/final_eval_perclass.csv` | P |
| scFoundation (full FT) | `runs_scfa_ft/ild/B/final_eval_perclass.csv` | B |
| scFoundation (full FT) | `runs_scfa_ft/ild/P2BA/final_eval_perclass.csv` | P2BA |
| scFoundation (full FT) | `runs_scfa_ft/ild/P2BJ/final_eval_perclass.csv` | P2BJ |
| scFoundation (full FT) | `runs_scfa_ft/ild/P2BU/final_eval_perclass.csv` | P2BU |
| scFoundation (full FT) | `runs_scfa_ft/ild/P2DS/final_eval_perclass.csv` | P2DS |

## Provenance warnings

- `VOID_premaskfix_perclass_scgpt_*.csv` are the scGPT numbers produced
  BEFORE the eval padding-mask fix. They are wrong. Kept only as the
  before half of the before/after. Do not quote them.
- `runs_scgpt/` (as opposed to `runs_scgpt_fixed/`) is likewise pre-fix.
- Everything is seed 0, single seed, no confidence intervals.

## Regenerating what was left behind

```
python pipeline/pilot/pilot_joint_balance.py --cohort {crc,aida,ild}   # bj h5ad + tokenized set
sbatch pipeline/pilot/pilot_scgpt_all.sbatch                           # scGPT arms + per-cell-type
sbatch pipeline/pilot/pilot_scfoundation_aligned.sbatch                # scFoundation frozen
sbatch pipeline/pilot/pilot_scfa_finetune.sbatch                       # scFoundation full fine-tune
```
