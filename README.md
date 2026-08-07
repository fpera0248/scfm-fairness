# Demographic Bias in Single-Cell Foundation Models

> **Active successor repo.** This repository continues the single-cell foundation-model fairness work and is where new methods land. It was duplicated with full history from [`scfm-bias-asi2026`](https://github.com/fpera0248/scfm-bias-asi2026), the frozen, citable artifact of the ACM-BCB ASI 2026 paper. To cite or reproduce the published paper, use that repo and its `ghcr.io/fpera0248/scfm-*` images; CI here publishes `scfm-fairness-*` images so the paper's artifacts stay untouched.

Code to reproduce *Demographic Bias in Single-Cell Foundation Models: Evaluation and Mitigation Across Models and Cohorts* (ACM-BCB ASI 2026 workshop). We measure how well scFoundation, Geneformer, and scGPT recover cell types for underrepresented demographic groups, and we test scDesign3 synthetic augmentation as a mitigation.

Cohorts: interstitial lung disease (ILD, ethnicity imbalance 127:1), colorectal cancer (CRC, 17:1), and the Asian Immune Diversity Atlas (AIDA, 8.4:1). Demographic axes: ethnicity (main analysis), sex and age (supplementary). Metrics: iLISI for embedding mixing, worst-group cell-type macro-F1 for recovery.

## What this repo contains

This repo holds the pipeline code and the pinned conda environment specs for the three models plus the scDesign3 augmentation step. It does not host the datasets or the model weights, which each carry their own license and access terms. Reproducing the numbers needs a Linux machine with an NVIDIA GPU, the three datasets, and the three model weight sets.

## Reproduce our results (recommended path)

The turnkey way to recreate our numbers is the **prebuilt container** — no environment setup, no path editing, one command per workflow:

```
# all-in-one image (every model + scDesign3); per-model images (scfm-scfoundation, …) also work
docker pull ghcr.io/fpera0248/scfm-all:latest
docker run --gpus all -v "$PWD/data":/data \
    ghcr.io/fpera0248/scfm-all:latest \
    reproduce <model> <cohort> <demographic>
```

> **Budget an hour or so for the first pull — these images are large.** Each bundles a foundation model **plus its weights and the scDesign3 R environment**, so they run **~13–15 GB**. The one-time `docker pull` is several minutes on a fast connection; on **HPC with Apptainer**, building the `.sif` from the OCI layers is the slow step and can take **the better part of an hour on shared/Lustre scratch** (much faster pointing `APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` at local/SSD scratch). It's a one-time, cached image-prep step — not the reproduction run itself — so plan for it once.

- `<model>` — `scfoundation` | `geneformer` | `scgpt`
- `<cohort>` — `ild` | `crc` | `aida`
- `<demographic>` — `ethnicity` | `sex` | `age`  *(AIDA and CRC are sex-balanced, so no `sex`)*

`reproduce` downloads the cohort from CZ CELLxGENE, wires the baked model checkpoint into the paths the scripts expect, and runs the full chain (extract → scDesign3 augment → embed → benchmark → downstream), writing outputs and figures under `/data`. On HPC, use Apptainer: `apptainer run --nv -B /your/data:/data <image>.sif reproduce <model> <cohort> <demographic>`.

That one command reproduces any of the **nine model×cohort combinations** on the ethnicity axis (the main-text results), each validated end to end in the container. The `sex`/`age` supplementary axes and the CRC full-prep path exist in the scripts but aren't yet wired for a turnkey run. **Full details — image list, GPU vs. CPU, and the exact boundary of what's verified-turnkey vs. what needs new code — are in [CONTAINER.md](CONTAINER.md).**

Everything below is for **understanding the method or adapting it to your own data/models** — you do not need it to reproduce our results.

## Understanding the pipeline (and adapting it)

To learn how the pipeline works, or to run it by hand and adapt it, follow one workflow end to end.

The reference workflow is `scfoundation/augmentedv4/ethnicity_scfoundation_workflow`. It runs scFoundation on the ILD cohort along the ethnicity axis, the main-text setup. Read "The pipeline, stage by stage" below, then open that folder and read its `step*.py` scripts in the order listed under Running a workflow. Every stage of the analysis lives there.

The demographic axis changes little across the three. The ethnicity workflow is the reference and groups cells by `self_reported_ethnicity`. The sex workflow is identical, grouping by `sex`. The age workflow adds one step, `step0a`, which extracts raw counts and bins age; ethnicity and sex reuse the existing raw counts and skip it.

The remaining workflows run the same pipeline for the other two models and the other two cohorts. Only the embedding stage differs by model, and the data files and disease-column setup differ by cohort. Use them for full reproduction. You do not need to read all of them to understand the method.

## Requirements

- Linux with an NVIDIA GPU. The original runs used Brown's Oscar HPC cluster.
- conda or miniforge. The original runs used miniforge3/25.3.0-3.
- The three datasets (see Data).
- The three model weight sets (see Model setup).

## The pipeline, stage by stage

Every workflow runs the same stages. Read this section to understand what each stage measures and why it matters, then use Running a workflow for the commands.

**1. Load and label.** Read the `.h5ad`, assign each cell to its demographic group from the `obs` column, and drop cells whose label is missing or a placeholder (`"nan"`, `"unknown"`). The analysis compares groups, so a pooled or mislabeled group corrupts every downstream number.

**2. Build the four rebalancing conditions.** From the loaded data, construct four versions:
- **Proportional:** the native class balance, untouched. Shows the bias as it exists.
- **BalancedAugmented:** scDesign3 generates synthetic minority cells until each group reaches the majority count.
- **BalancedUpsampled:** duplicate existing minority cells to that same target count.
- **Downsampled:** subsample every group down to the smallest group's count.

Each condition isolates a different mechanism: proportional measures the problem, augmentation adds new cells, upsampling adds copies, downsampling removes cells. Comparing them separates a real fix from an artifact.

**3. Generate synthetic cells with scDesign3.** Fit per-gene marginal distributions and the gene-gene correlation structure on the real minority cells, then simulate new cells in expression space. The synthetic cells pass through the same frozen encoder as real cells, so they have to look like real expression, not like points shifted in embedding space. Fitting the correlation structure is what keeps them from collapsing to noise.

**4. Embed with the frozen foundation model.** Pass each condition through scFoundation, Geneformer, or scGPT without fine-tuning, and keep the cell embeddings. These models get deployed as-is and reused across tasks, so freezing them measures the bias a downstream user actually inherits. This is the only stage that differs across the three models.

**5. Measure mixing with iLISI.** For each cell's local neighborhood in the embedding, measure how well demographic groups interleave: 1 is full mixing, 0 is full separation. A minority group sitting in its own region means the model encoded demographic structure, which is the bias signal.

**6. Validate the synthetic cells.** Run KS tests comparing synthetic against real cells on library size and gene sparsity. Augmentation only counts as a fix if the synthetic cells match real ones. If they differ, a high downstream score is memorization of an artifact, not recovery.

**7. Classify cell type on the frozen embeddings.** Train logistic regression and random forest on the embeddings to predict cell type. Running both matters: a result that holds under only one classifier is a classifier artifact, and one that holds under both is a property of the embedding.

**8. Trace learning curves.** Sweep the training-set size and record validation macro-F1 at each size. A curve that climbs steadily is genuine learning. A curve that jumps and then flatlines is memorization of duplicated cells. This is how upsampling's inflated internal score gets exposed.

**9. Run external validation and worst-group macro-F1.** Evaluate on a shared held-out set the model never trained on, compute macro-F1 for the smallest minority group, and bootstrap it over 1000 resamples for a confidence interval. Internal cross-validation leaks duplicated cells, and an overall score hides the group that fails, so the worst-group score on held-out data is the honest measure of who the model leaves behind.

**10. Classify disease (ILD and CRC only).** As a secondary check, predict normal versus disease and compute worst-group disease accuracy. This confirms the model ranking is not specific to cell-type labels. The smallest minorities have too few cells for a reliable per-group disease number, so treat it as directional, not a headline result. AIDA donors are all healthy, so this stage does not run on AIDA.

## Repo layout

One folder per (model, cohort, demographic) combination. The ILD root differs by model, so the pattern is not uniform. The exact roots:

**ILD**
```
scfoundation/augmentedv4/{demographic}_scfoundation_workflow
Geneformer/augmented/{demographic}_Geneformer_workflow
scGPT/{demographic}_scGPT_workflow
```

**CRC**
```
{model}/augmented_CRC/{demographic}_{model}_workflow
```

**AIDA**
```
{model}/augmented_AIDA/{demographic}_{model}_workflow
```

`{demographic}` is `age`, `ethnicity`, or `sex`. `{model}` is `scfoundation`, `Geneformer`, or `scGPT`. That gives 24 workflows: three models times three cohorts times three demographics, minus the three AIDA sex workflows.

AIDA has no sex workflow for any model. AIDA is sex-balanced, so the sex analysis does not apply and the paper does not report it. This is by design, not a missing file.

Ethnicity workflows produce the main-text results. Sex and age workflows produce the supplementary results.

## Model setup

The pipeline uses four conda environments: `scfoundation_gpu`, `geneformer310`, `scgpt310`, and `scdesign3_env` (the R-based scDesign3 stage). Their dependencies conflict, so keep them separate.

**`ENVIRONMENTS.md`** is the full rebuild guide: installing conda if you do not have it, creating each environment from its spec (slim `environment_<name>.yml` or fully pinned `environment_<name>.full.yml`), and the two that need an extra step — scGPT installs from pip (`scgpt==0.2.1`), and scDesign3 installs from source (version 1.5.0, via `devtools`). **`install.sh`** runs the whole thing in one pass:
```
bash install.sh          # slim specs (recommended)
bash install.sh --full   # fully pinned specs, exact Linux/CUDA reproduction
```

The environments provide the code to run each model. The pretrained model **weights** are separate downloads with their own licenses and are not in this repo:
- **scFoundation** — no manual download. The env's `modelgenerator==0.1.3` (genbio-ai) auto-pulls the checkpoint from the HF Hub repo `genbio-ai/scFoundation` (commit `cb434153`, `models.ckpt`, ~1.43 GB) on first use.
- **Geneformer** — HF repo `ctheodoris/Geneformer` (commit `fcd26c4`), installed as an editable package (`geneformer==0.1.0`). The 104M-model token dictionary `token_dictionary_gc104M.pkl` ships inside that repo.
- **scGPT** — package `scgpt==0.2.1` (bowang-lab/scGPT). The pretrained **whole-human** checkpoint (`scGPT_human/`: `best_model.pt`, `vocab.json`, `args.json`) is a separate download from the authors (Google Drive/Figshare, linked in their README); `fetch_weights.sh` retrieves it.

Pin the exact checkpoint for each model. A different checkpoint produces different embeddings and will not reproduce the reported numbers.

## Data

The three cohorts come from CZ CELLxGENE. This repo ships download commands, not the files, since CELLxGENE redistributes them under their own licenses. `fetch_data.sh` pulls all three into `data/`.

- ILD: Natri et al. 2024, Nature Genetics. GEO GSE227136, doi 10.1038/s41588-024-01702-0.
  https://datasets.cellxgene.cziscience.com/c3d9262e-0dc5-4eca-bf20-56e6d96d0306.h5ad
- CRC: Moorman et al. 2024, Nature. doi 10.1038/s41586-024-08560-0, epithelial compartment.
  https://datasets.cellxgene.cziscience.com/66cadf3b-4c71-4930-8add-fa748745704d.h5ad
- AIDA: Kock et al. 2025, Cell. Phase 1 Data Freeze v2.
  https://datasets.cellxgene.cziscience.com/f89a12c2-7a3b-415b-ab87-bbc550fe17f4.h5ad

These URLs point at the raw published objects. The pipeline filters them, so the raw cell counts do not match the post-QC counts in the paper. The raw AIDA v2 object holds 1,265,624 cells; the pipeline uses 1,165,872. Each dataset keeps its own CELLxGENE license, so check the terms before redistributing or publishing derived data.

## Input format

Each workflow reads an AnnData `.h5ad`. To run on your own data, the file needs:

- an expression matrix in `X`,
- an `obs` cell-type column (`cell_type`),
- an `obs` demographic column for the axis you run: `self_reported_ethnicity`, `sex`, or the age field.

Two rules the pipeline depends on:

- Synthetic cells carry the literal string `"nan"` in `self_reported_ethnicity`, `disease`, and `sex`. Exclude `"nan"` from every per-group count. Do not fold it into a real group.
- The age workflow uses `"unknown"` as an ethnicity value. Exclude it the same way.

Each model expects its own gene panel and tokenization. Preprocess your `.h5ad` to the model's expected input before the embedding stage, or the model will not produce valid representations.

## Running a workflow by hand (manual / adapting)

*You only need this to adapt the pipeline or run it outside the container — to reproduce our results, use the `reproduce` command above.* Run the `step*.py` scripts in numerical order from inside a workflow folder. The sequence:

- `step0a`: extract raw counts. Present only in age workflows; the other axes reuse its output.
- `step0b`: scDesign3 synthetic generation, written in R, run as stages 1 through 3.
- `step2a`: embed the cells with the frozen foundation model.
- `step3a`: benchmark.
- `step3b`: label propagation.
- `step4`: external validation. Computes worst-group macro-F1 on the held-out set.
- `step4a`: downstream classification.
- `step4b`: robustness stress tests.
- `step5`: learning curves.
- `step6`: per-group diagnostics.
- `step7`: representation diagnostics.
- `step8`: disease classification (ILD and CRC).
- `step9`: visualizations.

This is the sequence in the reference workflow; the core pipeline uses `step2a`, `step3a`, and `step3b` rather than standalone `step2` or `step3`. Some other workflows carry additional diagnostic or validation scripts around this core (for example `step0c` external validation in the age and sex workflows, `step1a` integrity checks, `step3d`, and `step7b`). They sort into place by name and are optional extras, not new required stages.

**Paths — read this before running.** The scripts use `/data` as a placeholder root for
every input, output, model, and log path (near the top of each script, e.g. the `BASE`
variable). It is *not* a real location on your machine — you set it. Two ways:

- **In a container:** mount your data at `/data` (`-v /your/data:/data` for Docker,
  `-B /your/data:/data` for Apptainer) and the paths resolve as-is.
- **On your own machine/cluster:** edit the `/data/...` path near the top of the script
  to your actual location. This release has no single config file that sets the path once,
  so set it per script (the paths are all under `/data`, so a find-replace of `/data` to
  your root works).

Seeding uses `numpy.random.default_rng` with a fixed `RANDOM_STATE` defined near the top of each script.

## Reproducing across the three models

The stages above are identical across scFoundation, Geneformer, and scGPT. Only stage 4, the embedding, differs. To reproduce all three, run the matching workflow for each. The iLISI, classification, learning-curve, and worst-group scripts run the same logic on whatever embeddings stage 4 produced.

## Auditing a new foundation model

The evaluation stages (5 through 10) operate on cell embeddings and do not depend on which model produced them, so the audit generalizes to any frozen single-cell model that maps cells to a fixed-length embedding.

The scripts are per-model copies, not a plugin interface. To audit a new model today:

1. Copy an existing workflow folder.
2. Replace the embedding stage (`step2a`) with your model's `h5ad` to embedding call, keeping the same output format (one embedding row per cell, aligned to `obs`).
3. Run the downstream stages unchanged.

## Data-quality notes

- Ethnicity values are cased inconsistently across cohorts. Lowercase-normalize before any per-group count, or a group splits into two buckets.
- `"nan"` (synthetic cells) and `"unknown"` (age workflow) are always exclusions, never a group.

## Citation

If you use this code or reproduce our results, please cite the paper:

> Peralta Castro, F., Zhang, J., & Singh, R. (2026). *Demographic Bias in Single-Cell Foundation Models: Evaluation and Mitigation Across Models and Cohorts.* In 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB Companion '26) — 4th Workshop on Advances in Systems Immunology (ASI 2026), Rende (CS), Italy. ACM. https://doi.org/10.1145/3807502.3833077

```bibtex
@inproceedings{peraltacastro2026demographic,
  author    = {Peralta Castro, Fernando and Zhang, Jiaqi and Singh, Ritambhara},
  title     = {Demographic Bias in Single-Cell Foundation Models: Evaluation and Mitigation Across Models and Cohorts},
  year      = {2026},
  booktitle = {17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB Companion '26)},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Rende (CS), Italy},
  doi       = {10.1145/3807502.3833077},
  isbn      = {979-8-4007-2652-1}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included, so GitHub shows a **"Cite this repository"** button (APA/BibTeX export) in the sidebar.

### Papers this work builds on

Please also cite the models, datasets, and methods you rely on. The full reference list is in the paper; the works this repository directly uses are:

**Foundation models (audited)**
- scFoundation — Hao et al., *Large-scale foundation model on single-cell transcriptomics.* Nature Methods 21:1481–1491, 2024.
- scGPT — Cui et al., *scGPT: toward building a foundation model for single-cell multi-omics using generative AI.* Nature Methods 21:1470–1480, 2024.
- Geneformer — Theodoris et al., *Transfer learning enables predictions in network biology.* Nature 618:616–624, 2023.

**Datasets (CZ CELLxGENE; each under its own license)**
- ILD — Natri et al., *Cell type-specific and disease-associated eQTL in the human lung.* Nature Genetics 56:595–604, 2024. doi:10.1038/s41588-024-01702-0
- CRC — Moorman et al., *Progressive plasticity during colorectal cancer metastasis.* Nature 637:947–954, 2025. doi:10.1038/s41586-024-08560-0
- AIDA — Kock et al., *Asian diversity in human immune cells.* Cell 188:2288–2308, 2025.

**Methods and baselines**
- scDesign3 (synthetic augmentation) — Song et al., *scDesign3 generates realistic in silico data for multimodal single-cell and spatial omics.* Nature Biotechnology 42:247–252, 2024.
- scIB / iLISI (mixing metric) — Luecken et al., *Benchmarking atlas-level data integration in single-cell genomics.* Nature Methods 19:41–50, 2022.
- Harmony (integration baseline) — Korsunsky et al., *Fast, sensitive and accurate integration of single-cell data with Harmony.* Nature Methods 16:1289–1296, 2019.
- Related prior work — Willem et al., *Biases in machine-learning models of human single-cell data.* Nature Cell Biology 27:384–392, 2025.
