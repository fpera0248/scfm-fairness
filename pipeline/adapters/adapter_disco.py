#!/usr/bin/env python
"""DISCO v1 adapter -> part-shaped h5ads (net-new race-labeled samples).

Pulls the 14 net-new DISCO GSE projects (232 race-labeled samples /
~1.31M cells; registry/ethnicity_registry.csv DEDUP_STATUS=net-new — the 4
Census/SCP-duplicate projects GSE115469, GSE126030, GSE151302, SCP1671 are
hard-excluded), keeps healthy-only race-labeled samples, maps DISCO's
free-text `race` to the corpus macro-groups, and emits raw-count h5ad parts
(same obs/var shape conventions as adapter_scea.py / build_corpus.py stage B)
plus MANIFEST_disco.csv + disco_run_report.json.

Endpoints (live-verified 2026-08-09; see registry/v1_eligibility.md):
  raw counts   GET {API}/download/getRawH5/{project_id}/{sample_id}
               -> 10x CellRanger v3 .h5 (HDF5), readable with
               scanpy.read_10x_h5. Contains ONLY DISCO's QC-passing cells
               (probe GSM3856591: 100 barcodes == the 100 annotated cells).
  cell types   GET {API}/toolkit/getCellTypeSample?sampleId={sample_id}
               -> TSV: cell_id, sample_id, cell_type, umap_1, umap_2,
               cell_type_score. These are DISCO CELLiD *predictions*
               (provenance recorded as 'predicted:CELLiD'; toolkit
               convention: score >= 0.8 = high confidence). cell_id is the
               bare 10x barcode and matches the h5 obs_names 1:1.
  Host MUST be disco.bii.a-star.edu.sg — DISCOtoolkit's hardcoded
  immunesinglecell.org prefix 404s. urllib speaks HTTP/1.1, which also
  sidesteps the HTTP/2 mid-stream INTERNAL_ERRORs seen with curl.

GENE NAMESPACE (probe finding, GSE132338/GSM3856591, 2026-08-09): the h5's
features/id AND features/name are both gene SYMBOLS (0/33,538 ENSG; the 10x
GRCh38-3.0.0 reference set keyed by symbol). DISCO reprocesses every sample
through one pipeline, so expect symbols everywhere -> a --gene-map TSV
(symbol \\t ensembl_id, e.g. built from the CellRanger GRCh38-3.0.0
features.tsv.gz) is effectively REQUIRED for real runs. The adapter still
verifies the namespace per sample (loudly on the first one) and uses
Ensembl ids directly if a sample ever ships them.

Sample metadata comes from the versioned toolkit dump snapshot
(disco_sample_metadata.tsv beside this script; 5,940 samples, 2026-08-09).
If the file is missing and not --offline it is re-fetched live, but the live
endpoint now returns a truncated ~511-row page -- prefer the snapshot.

Outputs under --out:
  h5ad_parts/<project>/<sample>.h5ad   one part per sample (cells x genes,
                                       raw counts CSR float32; resumable:
                                       existing parts are skipped)
  MANIFEST_disco.csv                   one row per candidate sample
  disco_run_report.json                label mapping / exclusion / gene-
                                       namespace / annotation-coverage log

obs: dataset_id (GSE), sample_id, donor_id (subject_id, else sample_id),
donor_key='disco|<project>|<donor>', self_reported_ethnicity (verbatim
race), group, subgroup_uncertain, provenance='disco_race_freetext',
disease, sex, tissue, cell_type, cell_type_score, cell_type_provenance.
var: index=versionless Ensembl id ('gene_id'), column gene_symbol.

--dry-run: metadata accounting only (no downloads), manifest/report with
status='dry_run'.  --offline: cache-only, never touch the network.

Usage:
  python adapter_disco.py --out /oscar/scratch/fperalta/disco_corpus --dry-run
  python adapter_disco.py --out /oscar/scratch/fperalta/disco_corpus \\
      --gene-map grch38_3.0.0_symbol2ensg.tsv
"""
import argparse
import csv
import json
import pathlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

API_BASE = "https://disco.bii.a-star.edu.sg/disco_v3_api"
RAW_H5_URL = API_BASE + "/download/getRawH5/{project}/{sample}"
CELLTYPE_URL = API_BASE + "/toolkit/getCellTypeSample?sampleId={sample}"
METADATA_URL = API_BASE + "/toolkit/getSampleMetadata"

# Full toolkit dump snapshot (5,940 samples, fetched 2026-08-09), versioned
# next to this script. The live getSampleMetadata endpoint now returns a
# truncated ~511-row page, so the snapshot is authoritative for v1.
DEFAULT_METADATA = pathlib.Path(__file__).resolve().parent / "disco_sample_metadata.tsv"

PROVENANCE = "disco_race_freetext"
CELLTYPE_PROVENANCE = "predicted:CELLiD"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ENSG_RE = re.compile(r"^ENSG\d+")

# 14 net-new projects (registry/ethnicity_registry.csv, DISCO hub,
# DEDUP_STATUS=net-new). Census/SCP duplicates deliberately absent:
# GSE115469, GSE126030, GSE151302, SCP1671.
NET_NEW_PROJECTS = [
    "GSE114156", "GSE118184", "GSE129845", "GSE130228", "GSE130560",
    "GSE132338", "GSE136353", "GSE144136", "GSE147082", "GSE151528",
    "GSE156285", "GSE167960", "GSE171964", "GSE206265",
]

# ---------------------------------------------------------------------------
# race free-text -> macro-group. Keys are normalize_label()-normalized
# (lowercase, hyphens -> spaces, whitespace collapsed). Values:
# (group, subgroup_uncertain) or the sentinels EXCLUDE / UNLABELED.
# Covers every value observed in the net-new dump slice (2026-08-09) plus
# the compound values listed in registry/v1_eligibility.md.
EXCLUDE = "__EXCLUDE__"
UNLABELED = "__UNLABELED__"
LABEL_MAP = {
    # European (white/caucasian variants)
    "white": ("European", False),
    "caucasian": ("European", False),
    "white non hispanic": ("European", False),
    "non hispanic white": ("European", False),
    "non hispanic whites": ("European", False),
    # African
    "black": ("African", False),
    "black non hispanic": ("African", False),
    "african american": ("African", False),
    "america african": ("African", False),
    "black or african american": ("African", False),
    # Hispanic / Latino
    "hispanic": ("HispanicLatino", False),
    "latino": ("HispanicLatino", False),
    "latin": ("HispanicLatino", False),
    "hispanic or latino": ("HispanicLatino", False),
    "white hispanic or latino": ("HispanicLatino", False),
    "multiple race hispanic or latino": ("HispanicLatino", False),
    # East / SE Asian
    "chinese": ("EastSEAsian", False),
    "chinese american": ("EastSEAsian", False),
    "east asian": ("EastSEAsian", False),
    # generic 'asian' -> East/SE Asian with subgroup_uncertain
    "asian": ("EastSEAsian", True),
    "asian non hispanic": ("EastSEAsian", True),
    # South Asian
    "south asian": ("SouthAsian", False),
    "indian": ("SouthAsian", False),
    "asian indian": ("SouthAsian", False),
    # Middle Eastern
    "armenian": ("MiddleEastern", False),
    # Native American
    "native american": ("NativeAmerican", False),
    "native": ("NativeAmerican", False),
    # excluded, logged
    "non hispanic": EXCLUDE,          # bare ethnicity-only label
    "mixed": EXCLUDE,
    "multiple race": EXCLUDE,
    "pacific islander": EXCLUDE,
    # missing / placeholder
    "": UNLABELED,
    "na": UNLABELED,
    "n/a": UNLABELED,
    "nan": UNLABELED,
    "null": UNLABELED,
    "none": UNLABELED,
    "unknown": UNLABELED,
    "not available": UNLABELED,
}

HEALTHY_DISEASE = {"control", "healthy", "normal", "none", "ctrl"}
UNKNOWN_DISEASE = {"", "na", "n/a", "nan", "null", "unknown", "not available"}


def normalize_label(raw):
    s = re.sub(r"\s+", " ", str(raw).replace("-", " ")).strip().lower()
    return s


def log(msg):
    print(f"[adapter_disco] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fetch helpers (atomic, cached; urllib = HTTP/1.1 which DISCO needs)


def fetch(url, dest, offline=False, retries=3):
    """Download url -> dest (atomic, cached). Returns dest or None."""
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scfm-adapter/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - retry then surface
            log(f"fetch attempt {attempt + 1}/{retries} failed for {url}: {e}")
    if tmp.exists():
        tmp.unlink()
    return None


def fetch_resumable(url, dest, offline=False, max_rounds=800):
    """Range-resume download for DISCO's flaky file endpoints.

    The server silently closes large transfers after ~0.1-1 MB (verified
    2026-08-09: 20 MB h5s arrive truncated at random offsets on every node
    AND from outside the cluster) but it honors Range requests, so we
    reconnect and append until it answers 416 (= past EOF, file complete).
    The .part file survives failed runs, so progress is never lost.
    """
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    stall = 0
    for round_ in range(max_rounds):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "scfm-adapter/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                if have and getattr(r, "status", 200) == 200:
                    tmp.unlink()  # server ignored Range -> restart clean
                    have = 0
                with open(tmp, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # requested range past EOF: complete
                tmp.rename(dest)
                return dest
            log(f"resume round {round_ + 1} HTTP {e.code} for {url}")
        except Exception:  # noqa: BLE001 - expected mid-stream drops; keep going
            pass
        new = tmp.stat().st_size if tmp.exists() else 0
        stall = stall + 1 if new == have else 0
        if stall >= 10:
            log(f"resume: no progress after {stall} rounds "
                f"({new:,} bytes) for {url}; keeping .part for rerun")
            return None
        if round_ % 50 == 49:
            log(f"resume {dest.name}: {new:,} bytes after {round_ + 1} rounds")
        time.sleep(0.5)
    log(f"resume: round budget exhausted for {url}; keeping .part for rerun")
    return None


def fetch_h5(project, sample, cache, offline):
    """Fetch the raw 10x h5 and sanity-check the HDF5 magic. A corrupt cached
    file (mid-stream abort) is deleted and refetched once."""
    dest = cache / project / f"{sample}.h5"
    for _ in range(2):
        p = fetch_resumable(RAW_H5_URL.format(project=project, sample=sample),
                            dest, offline)
        if p is None:
            return None
        with open(p, "rb") as f:
            if f.read(8) == HDF5_MAGIC:
                return p
        log(f"{sample}: cached h5 is not HDF5 (truncated/error body) -> refetch")
        p.unlink()
        if offline:
            return None
    return None


def fetch_celltypes(sample, cache, offline):
    """Fetch CELLiD annotations -> {barcode: (cell_type, score)} or None.
    Handles the no-annotation case (HTTP error / non-TSV body) gracefully."""
    dest = cache / "celltype" / f"{sample}.tsv"
    p = fetch(CELLTYPE_URL.format(sample=sample), dest, offline)
    if p is None:
        return None
    ann = {}
    with open(p, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        if "cell_id" not in header or "cell_type" not in header:
            log(f"{sample}: cell-type endpoint returned non-TSV body -> no annotations")
            p.unlink()  # do not cache garbage
            return None
        i_id, i_ct = header.index("cell_id"), header.index("cell_type")
        i_sc = header.index("cell_type_score") if "cell_type_score" in header else None
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) <= i_ct:
                continue
            score = np.nan
            if i_sc is not None and len(row) > i_sc:
                try:
                    score = float(row[i_sc])
                except ValueError:
                    pass
            ann.setdefault(row[i_id], (row[i_ct], score))
    return ann


# ---------------------------------------------------------------------------
# Sample metadata (toolkit dump; TSV)


def load_sample_metadata(path, cache, offline):
    path = pathlib.Path(path)
    if not path.exists():
        log(f"metadata file {path} missing -> fetching from {METADATA_URL}")
        path = fetch(METADATA_URL, cache / "sample_metadata.tsv", offline)
        if path is None:
            return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_label(raw, project, report):
    """Map a verbatim race label. Returns (group, uncertain), EXCLUDE,
    UNLABELED, or None (unmapped -> NEED_REVIEW)."""
    hit = LABEL_MAP.get(normalize_label(raw))
    if hit is None:
        report["unmapped_NEED_REVIEW"][f"{project} :: {raw}"] += 1
        return None
    if hit == EXCLUDE:
        report["excluded_labels"][f"{project} :: {raw}"] += 1
    elif hit == UNLABELED:
        report["unlabeled"][f"{project} :: {raw or '<empty>'}"] += 1
    return hit


def disease_status(disease, keep_unknown):
    d = normalize_label(disease)
    if d in HEALTHY_DISEASE:
        return "healthy"
    if d in UNKNOWN_DISEASE:
        return "unknown_kept" if keep_unknown else "unknown_dropped"
    return "disease"


def select_samples(meta_rows, projects, only_samples, args, report):
    """Apply race + healthy-only policy. Returns (kept, manifest_rows) where
    kept = list of per-sample dicts ready for download/writing."""
    kept, rows = [], []
    for m in meta_rows:
        project = m.get("project_id", "")
        sample = m.get("sample_id", "")
        if project not in projects:
            continue
        if only_samples and sample not in only_samples:
            continue
        race = (m.get("race") or "").strip()
        row = {"project_id": project, "sample_id": sample, "race_verbatim": race,
               "group": "", "subgroup_uncertain": "", "donor_id": "",
               "disease": m.get("disease", ""), "sex": m.get("gender", ""),
               "tissue": m.get("tissue", ""),
               "cell_number_meta": m.get("cell_number", ""),
               "status": "", "n_cells_written": "", "n_cells_annotated": "",
               "n_genes_out": "", "gene_namespace": "", "gene_mapped_frac": "",
               "part_file": ""}
        resolved = resolve_label(race, project, report)
        if resolved is None:
            row["status"] = "dropped_unmapped_NEED_REVIEW"
            rows.append(row)
            continue
        if resolved == EXCLUDE:
            row["status"] = "dropped_excluded_label"
            rows.append(row)
            continue
        if resolved == UNLABELED:
            row["status"] = "dropped_unlabeled"
            rows.append(row)
            continue
        dstat = disease_status(m.get("disease", ""), args.keep_unknown_disease)
        if dstat == "disease":
            row["status"] = "dropped_disease"
            report["dropped_disease_values"][m.get("disease", "")] += 1
            rows.append(row)
            continue
        if dstat == "unknown_dropped":
            row["status"] = "dropped_disease_unknown"
            report["dropped_disease_unknown"] += 1
            rows.append(row)
            continue
        group, uncertain = resolved
        subject = (m.get("subject_id") or "").strip()
        donor = subject if subject and subject.upper() not in ("NA", "N/A") else sample
        row.update(group=group, subgroup_uncertain=uncertain, donor_id=donor,
                   status="selected")
        rows.append(row)
        kept.append({"row": row, "project": project, "sample": sample,
                     "donor": donor, "race": race, "group": group,
                     "uncertain": uncertain,
                     "disease": m.get("disease", ""),
                     "disease_status": dstat,
                     "sex": m.get("gender", ""), "tissue": m.get("tissue", "")})
    return kept, rows


# ---------------------------------------------------------------------------
# Genes


def load_gene_map(path):
    """TSV symbol -> Ensembl ID (versionless). Header optional."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sym, ens = parts[0].strip(), parts[1].strip()
            if ens.startswith("ENSG"):
                mapping[sym.upper()] = ens.split(".")[0]
    return mapping


class GeneMapRequired(RuntimeError):
    pass


def resolve_gene_space(ids, gene_map):
    """ids -> (namespace, ensembl list with None for unmappable, mapped_frac)."""
    n = max(len(ids), 1)
    ens_frac = sum(bool(ENSG_RE.match(g)) for g in ids) / n
    if ens_frac >= 0.9:
        ens = [g.split(".")[0] if ENSG_RE.match(g) else None for g in ids]
        return "ensembl", ens, ens_frac
    if gene_map is None:
        raise GeneMapRequired(
            f"gene ids are symbols (ENSG fraction {ens_frac:.1%}); "
            "pass --gene-map TSV (symbol\\tensembl_id, e.g. from the "
            "CellRanger GRCh38-3.0.0 features.tsv.gz)")
    ens = [gene_map.get(g.upper()) for g in ids]
    return "symbols_mapped", ens, sum(e is not None for e in ens) / n


def collapse_by_ensembl(X, ens_ids, symbols):
    """Drop unmapped columns, collapse duplicate Ensembl ids by sum.
    Returns (X csr float32, uniq_ens, symbol per uniq_ens)."""
    import scipy.sparse

    keep = np.array([e is not None for e in ens_ids])
    X = X[:, np.flatnonzero(keep)]
    ids = np.array([e for e in ens_ids if e is not None])
    syms = [s for s, k in zip(symbols, keep) if k]
    first_sym = {}
    for e, s in zip(ids, syms):
        first_sym.setdefault(e, s)
    uniq, inv = np.unique(ids, return_inverse=True)
    if len(uniq) < len(ids):
        M = scipy.sparse.csr_matrix(
            (np.ones(len(inv), dtype=np.float32), (np.arange(len(inv)), inv)),
            shape=(len(inv), len(uniq)))
        X = X @ M
    else:
        X = X[:, np.argsort(ids)]
    return (scipy.sparse.csr_matrix(X, dtype=np.float32), uniq,
            [first_sym[e] for e in uniq])


# ---------------------------------------------------------------------------
# Per-sample pipeline


def process_sample(item, args, gene_map, report, first_check):
    """Download + convert one sample. Mutates item['row']; returns None."""
    import anndata
    import pandas as pd
    import scanpy as sc

    row, project, sample = item["row"], item["project"], item["sample"]
    safe = sample.replace("/", "_")
    out_fp = args.out / "h5ad_parts" / project / f"{safe}.h5ad"
    row["part_file"] = str(out_fp.relative_to(args.out))
    if out_fp.exists():
        row["status"] = "ok_cached"
        log(f"{sample}: {out_fp.name} exists, skipping")
        return

    h5 = fetch_h5(project, sample, args.cache, args.offline)
    if h5 is None:
        row["status"] = "ERROR_no_h5"
        log(f"{sample}: raw h5 unavailable")
        return
    try:
        ad0 = sc.read_10x_h5(h5)
    except Exception as e:  # noqa: BLE001 - corrupt cache -> drop it
        h5.unlink(missing_ok=True)
        row["status"] = "ERROR_h5_unreadable"
        log(f"{sample}: h5 unreadable ({e!r}); cache entry deleted -> rerun refetches")
        return

    # CITE-seq projects (e.g. GSE171964) may carry Antibody Capture features
    if "feature_types" in ad0.var.columns:
        gex = (ad0.var["feature_types"].astype(str) == "Gene Expression").values
        if not gex.all():
            report["per_project"][project]["non_gex_features_dropped"] = \
                int((~gex).sum())
            ad0 = ad0[:, gex]

    ids = [str(g) for g in (ad0.var["gene_ids"] if "gene_ids" in ad0.var.columns
                            else ad0.var_names)]
    namespace, ens, frac = resolve_gene_space(ids, gene_map)
    row["gene_namespace"], row["gene_mapped_frac"] = namespace, round(frac, 4)
    if first_check["pending"]:
        first_check["pending"] = False
        log("=" * 72)
        log(f"GENE NAMESPACE CHECK (first sample {project}/{sample}):")
        log(f"  namespace={namespace}  ENSG-or-mapped fraction={frac:.1%}  "
            f"n_features={len(ids)}")
        if namespace == "symbols_mapped":
            log("  DISCO ships gene SYMBOLS in features/id -> mapping via --gene-map")
        log("=" * 72)
        report["gene_namespace_first_sample"] = {
            "project": project, "sample": sample, "namespace": namespace,
            "mapped_fraction": round(frac, 4), "n_features": len(ids)}
    if namespace == "symbols_mapped" and frac < 0.5:
        row["status"] = "ERROR_gene_map_coverage"
        log(f"{sample}: gene map covers only {frac:.1%} of symbols -- refusing")
        return

    X, uniq_ens, uniq_sym = collapse_by_ensembl(
        ad0.X, ens, [str(s) for s in ad0.var_names])
    row["n_genes_out"] = len(uniq_ens)

    ann = fetch_celltypes(sample, args.cache, args.offline)
    barcodes = [str(b) for b in ad0.obs_names]
    if ann is None:
        cts, scores = [""] * len(barcodes), [np.nan] * len(barcodes)
        n_ann = 0
        report["per_project"][project]["samples_without_celltype_tsv"] = \
            report["per_project"][project].get("samples_without_celltype_tsv", 0) + 1
    else:
        cts = [ann.get(b, ("", np.nan))[0] for b in barcodes]
        scores = [ann.get(b, ("", np.nan))[1] for b in barcodes]
        n_ann = sum(bool(c) for c in cts)
        if n_ann < len(barcodes):
            log(f"{sample}: {len(barcodes) - n_ann}/{len(barcodes)} h5 cells "
                f"lack a CELLiD annotation")
    row["n_cells_annotated"] = n_ann

    obs = pd.DataFrame({
        "dataset_id": project,
        "sample_id": sample,
        "donor_id": item["donor"],
        "donor_key": f"disco|{project}|{item['donor']}",
        "self_reported_ethnicity": item["race"],
        "group": item["group"],
        "subgroup_uncertain": item["uncertain"],
        "provenance": PROVENANCE,
        "disease": (item["disease"] if item["disease_status"] == "healthy"
                    else "unknown"),
        "sex": item["sex"],
        "tissue": item["tissue"],
        "cell_type": cts,
        "cell_type_score": np.array(scores, dtype=np.float32),
        "cell_type_provenance": [CELLTYPE_PROVENANCE if c else "" for c in cts],
    }, index=pd.Index([f"{sample}|{b}" for b in barcodes], name="cell_id"))
    var = pd.DataFrame({"gene_symbol": uniq_sym},
                       index=pd.Index(uniq_ens, name="gene_id"))
    ad = anndata.AnnData(X=X, obs=obs, var=var)
    ad.uns["source"] = "DISCO"
    ad.uns["source_url"] = RAW_H5_URL.format(project=project, sample=sample)
    ad.uns["counts_layer"] = "raw_10x_h5"
    ad.uns["gene_namespace"] = namespace
    ad.uns["cell_type_provenance"] = CELLTYPE_PROVENANCE

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fp.with_suffix(".h5ad.tmp")
    ad.write_h5ad(tmp)
    tmp.rename(out_fp)
    row["status"] = "ok"
    row["n_cells_written"] = ad.n_obs
    log(f"{sample}: wrote {out_fp.name} ({ad.n_obs:,} cells x {ad.n_vars:,} genes)")


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="output dir (h5ad_parts/, MANIFEST_disco.csv, report)")
    ap.add_argument("--cache", type=pathlib.Path, default=None,
                    help="download cache (default: <out>/cache)")
    ap.add_argument("--metadata", type=pathlib.Path, default=DEFAULT_METADATA,
                    help="toolkit sample-metadata dump (TSV; default: cached "
                         "scratchpad copy, refetched live if missing)")
    ap.add_argument("--gene-map", type=pathlib.Path, default=None,
                    help="TSV symbol->Ensembl for symbol var names (required "
                         "in practice: DISCO h5s ship symbols; probe 2026-08-09)")
    ap.add_argument("--projects", type=str, default=None,
                    help="comma-separated project subset (default: all 14 net-new)")
    ap.add_argument("--samples", type=str, default=None,
                    help="comma-separated sample_id subset (for probing)")
    ap.add_argument("--keep-unknown-disease", action="store_true",
                    help="keep samples whose disease field is NA/empty "
                         "(default: drop them; healthy-only policy)")
    ap.add_argument("--dry-run", action="store_true",
                    help="metadata accounting only; no downloads, no h5ads")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; fail on cache misses")
    args = ap.parse_args(argv)

    args.cache = args.cache or args.out / "cache"
    projects = args.projects.split(",") if args.projects else NET_NEW_PROJECTS
    unknown = [p for p in projects if p not in NET_NEW_PROJECTS]
    if unknown:
        ap.error(f"not in the 14 net-new DISCO projects: {unknown} "
                 f"(Census/SCP duplicates GSE115469/GSE126030/GSE151302/"
                 f"SCP1671 are excluded by design)")
    only_samples = set(args.samples.split(",")) if args.samples else None

    meta = load_sample_metadata(args.metadata, args.cache, args.offline)
    if meta is None:
        log("FATAL: sample metadata unavailable")
        return 1
    gene_map = load_gene_map(args.gene_map) if args.gene_map else None
    if gene_map is not None:
        log(f"gene map: {len(gene_map):,} symbol->ENSG entries")

    report = {"adapter": "disco_v1", "dry_run": args.dry_run,
              "excluded_labels": Counter(), "unlabeled": Counter(),
              "unmapped_NEED_REVIEW": Counter(),
              "dropped_disease_values": Counter(),
              "dropped_disease_unknown": 0,
              "per_project": {p: {} for p in projects}}
    kept, rows = select_samples(meta, set(projects), only_samples, args, report)
    log(f"{len(rows)} candidate samples in scope; {len(kept)} pass "
        f"race+disease policy "
        f"(projected {sum(int(float(k['row']['cell_number_meta'] or 0)) for k in kept):,} cells)")

    first_check = {"pending": True}
    if args.dry_run:
        for item in kept:
            item["row"]["status"] = "dry_run"
    else:
        for item in kept:
            try:
                process_sample(item, args, gene_map, report, first_check)
            except GeneMapRequired as e:
                item["row"]["status"] = "ERROR_symbols_need_gene_map"
                log(f"FATAL: {e}")
                log("aborting run: DISCO's pipeline is uniform, remaining "
                    "samples would fail identically")
                break
            except Exception as e:  # noqa: BLE001 - keep going, record failure
                item["row"]["status"] = f"ERROR_{type(e).__name__}"
                log(f"{item['sample']}: FAILED: {e!r}")

    # per-project rollup
    for p in projects:
        prows = [r for r in rows if r["project_id"] == p]
        report["per_project"][p].update(
            n_candidates=len(prows),
            n_selected=sum(r["status"] in ("selected", "dry_run", "ok", "ok_cached")
                           or r["status"].startswith("ERROR") for r in prows),
            n_written=sum(r["status"] in ("ok", "ok_cached") for r in prows),
            groups=dict(sorted(Counter(
                r["group"] for r in prows if r["group"]).items())))

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["project_id", "sample_id", "status", "race_verbatim", "group",
              "subgroup_uncertain", "donor_id", "disease", "sex", "tissue",
              "cell_number_meta", "n_cells_written", "n_cells_annotated",
              "n_genes_out", "gene_namespace", "gene_mapped_frac", "part_file"]
    with open(args.out / "MANIFEST_disco.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for k in ("excluded_labels", "unlabeled", "unmapped_NEED_REVIEW",
              "dropped_disease_values"):
        report[k] = dict(sorted(report[k].items()))
    with open(args.out / "disco_run_report.json", "w") as f:
        json.dump(report, f, indent=1)

    log(f"manifest -> {args.out / 'MANIFEST_disco.csv'}")
    log(f"report   -> {args.out / 'disco_run_report.json'}")
    if report["unmapped_NEED_REVIEW"]:
        log(f"NEED_REVIEW unmapped labels: {report['unmapped_NEED_REVIEW']}")
    n_bad = sum(r["status"].startswith("ERROR") for r in rows)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
