#!/usr/bin/env python
"""EBI Single Cell Expression Atlas (SCEA) v1 adapter -> part-shaped h5ads.

Pulls the 18 human SCEA experiments that carry per-assay ethnicity fields
(inventory: final_hub_accounting.json, mined 2026-08 from condensed SDRFs),
joins per-assay donor + self-reported ethnicity + disease from the condensed
SDRF, maps free-text labels to the corpus macro-groups, keeps healthy-only
assays, and emits raw-count h5ad parts (same shape conventions as
build_corpus.py stage B / the HuBMAP adapter) plus MANIFEST_scea.csv.

Source layout (verified 2026-08-09 via FTP listing of
https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-GEOD-81608/):
  <EXP>.aggregated_counts.mtx.gz        raw aggregated counts, genes x cells (MatrixMarket)
  <EXP>.aggregated_counts.mtx_rows.gz   gene ids (2 tab cols, Ensembl id repeated)
  <EXP>.aggregated_counts.mtx_cols.gz   cell/assay ids (1 col)
  <EXP>.condensed-sdrf.tsv              accession \t array_design \t assay_id \t
                                        attr_class \t attr_name \t value [\t ontology_uri]
(.gz and plain variants both exist on the FTP; we prefer .gz and fall back.)

Assay-id join: SMART-like experiments have matrix cols == SDRF assay ids;
droplet experiments have cols like '<ASSAY>-<ACGTBARCODE>' -> prefix match.

Ethnicity attr names seen across the 18: 'ethnic group' (17), 'ancestry
category' (E-ENAD-27); matched by regex. Donor attr: 'individual' (fallback:
SDRF assay id, flagged). Genes are Ensembl (versions stripped, fraction of
ENSG-prefixed ids logged per experiment).

Outputs under --out:
  h5ad_parts/<EXP>.part_XXXX.h5ad   (<= --chunk cells each, raw counts,
                                     cells x genes CSR; resumable: existing
                                     parts are skipped)
  MANIFEST_scea.csv                 one row per experiment
  scea_run_report.json              per-label mapping/exclusion/unmapped log

--dry-run: fetch/parse SDRFs only (no matrix downloads), print the per-
experiment accounting, write manifest + report with status='dry_run'.

Usage:
  python adapter_scea.py --out /oscar/scratch/fperalta/scea_corpus --dry-run
  python adapter_scea.py --out /oscar/scratch/fperalta/scea_corpus
"""
import argparse
import csv
import gzip
import json
import pathlib
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import numpy as np

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments"

# The 18 human experiments with an ethnicity-like SDRF attribute
# (final_hub_accounting.json -> SCEA.per_experiment, 2026-08).
EXPERIMENTS = [
    "E-ANND-1", "E-ANND-3", "E-CURD-119", "E-CURD-7", "E-ENAD-21",
    "E-ENAD-27", "E-GEOD-139324", "E-GEOD-81547", "E-GEOD-81608",
    "E-GEOD-83139", "E-GEOD-99795", "E-HCAD-15", "E-HCAD-31", "E-HCAD-4",
    "E-HCAD-5", "E-HCAD-6", "E-MTAB-10287", "E-MTAB-9467",
]

PROVENANCE = "scea_sdrf_selfreported"
GROUPS = ["African", "EastSEAsian", "European", "HispanicLatino", "SouthAsian"]

ETHNICITY_ATTR_RE = re.compile(r"ethnic|ancestr|\brace\b", re.I)
DONOR_ATTRS = ["individual", "donor", "donor id", "individual identifier"]
DISEASE_ATTRS = ["disease", "disease state"]
NORMAL_DISEASE = {"normal", "healthy", "none"}
BARCODE_RE = re.compile(r"^[ACGTN]{6,}(-\d+)?$")
ENSG_RE = re.compile(r"^ENSG\d+")
VERSION_RE = re.compile(r"\.\d+$")

# ---------------------------------------------------------------------------
# Label -> macro-group mapping, seeded from final_hub_accounting.json values.
# Keys are normalize_label()-normalized. Values: (group, subgroup_uncertain)
# or the sentinels EXCLUDE / UNLABELED.
EXCLUDE = "__EXCLUDE__"
UNLABELED = "__UNLABELED__"
LABEL_MAP = {
    # European
    "white": ("European", False),
    "caucasian": ("European", False),
    "european": ("European", False),
    "european (white british)": ("European", False),
    # African
    "black": ("African", False),
    "african american": ("African", False),   # also covers 'African-american'
    # East / SE Asian
    "thai": ("EastSEAsian", False),
    "filipino": ("EastSEAsian", False),
    "asian": ("EastSEAsian", True),           # unspecified -> subgroup_uncertain
    # South Asian
    "asian indian": ("SouthAsian", False),
    # Hispanic / Latino
    "hispanic": ("HispanicLatino", False),
    "latino": ("HispanicLatino", False),
    "hispanic or latin american": ("HispanicLatino", False),
    # mixed / other -> excluded, logged
    "mixed south american": EXCLUDE,
    "mixed white/asian": EXCLUDE,
    "pacific islander": EXCLUDE,
    "african american, hispanic": EXCLUDE,    # E-HCAD-6 pooled-donor label
    # missing
    "": UNLABELED,
    "nan": UNLABELED,
    "not available": UNLABELED,
    "not applicable": UNLABELED,
    "unknown": UNLABELED,
    "na": UNLABELED,
    "n/a": UNLABELED,
    "none of these": UNLABELED,
}


def normalize_label(raw):
    s = re.sub(r"\s+", " ", str(raw)).strip().lower()
    s = re.sub(r"\s*-\s*", " ", s) if "american" in s else s  # African-american
    return s


def log(msg):
    print(f"[adapter_scea] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fetch helpers


THROTTLE = 1.0  # polite inter-request delay: EBI intermittently refuses
                # connections from clients that hammer it (observed 2026-08-09)


def fetch(url, dest, offline=False, retries=8):
    """Download url -> dest (atomic, cached). Returns dest or None.

    Refused/reset connections back off exponentially (EBI rate-limits);
    404/410 are permanent and give up immediately (feeds the sdrf.txt
    fallback without burning the retry budget).
    """
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        time.sleep(THROTTLE if attempt == 0 else min(90, 5 * 2 ** (attempt - 1)))
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                log(f"fetch: {url} -> HTTP {e.code} (permanent), giving up")
                break
            log(f"fetch attempt {attempt + 1}/{retries} failed for {url}: {e}")
        except Exception as e:  # noqa: BLE001 - retry then surface
            log(f"fetch attempt {attempt + 1}/{retries} failed for {url}: {e}")
    if tmp.exists():
        tmp.unlink()
    return None


def fetch_first(acc, suffixes, cache, offline=False):
    """Try each filename suffix for an experiment; return first cached path."""
    for suf in suffixes:
        p = fetch(f"{FTP_BASE}/{acc}/{acc}{suf}", cache / f"{acc}{suf}", offline)
        if p is not None:
            return p
    return None


def open_maybe_gz(path):
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SDRF


def parse_full_sdrf(path):
    """Fallback for experiments without a condensed SDRF: parse the wide
    <EXP>.sdrf.txt into the same {assay_id: {attr: value}} shape.

    attr names = lowercased Characteristics[...] / Factor Value[...] names
    (Characteristics wins when both exist, header order). Assay id preference
    mirrors what condensed SDRFs use: Comment[ENA_RUN] > Assay Name >
    Scan Name > Source Name. First value per assay wins (one row per file).
    """
    with open_maybe_gz(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return None
        attr_idx = {}
        for i, h in enumerate(header):
            m = re.match(r"(?:characteristics|factor ?value)\s*\[(.+)\]\s*$",
                         h.strip(), re.I)
            if m:
                attr_idx.setdefault(m.group(1).strip().lower(), i)
        low = [h.strip().lower() for h in header]
        id_cols = [low.index(c) for c in
                   ("comment[ena_run]", "assay name", "scan name", "source name")
                   if c in low]
        if not id_cols or not attr_idx:
            return None
        assays = defaultdict(dict)
        for row in reader:
            aid = next((row[i].strip() for i in id_cols
                        if i < len(row) and row[i].strip()), None)
            if not aid:
                continue
            for name, i in attr_idx.items():
                if i < len(row) and row[i].strip():
                    assays[aid].setdefault(name, row[i].strip())
    return dict(assays) or None


def load_sdrf(acc, sdrf_dir, cache, offline=False):
    """Parse condensed SDRF -> {assay_id: {attr_name: value}} (last one wins).

    Prefers a pre-downloaded copy in --sdrf-dir, else fetches into cache.
    Experiments without a condensed SDRF on the FTP (6 of the 18) fall back
    to the wide <EXP>.sdrf.txt.
    """
    fname = f"{acc}.condensed-sdrf.tsv"
    path = pathlib.Path(sdrf_dir) / fname if sdrf_dir else None
    if path is None or not path.exists():
        path = fetch(f"{FTP_BASE}/{acc}/{fname}", cache / fname, offline)
    if path is not None:
        assays = defaultdict(dict)
        with open_maybe_gz(path) as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 6:
                    continue
                assay, attr_name, value = row[2], row[4].strip().lower(), row[5].strip()
                assays[assay][attr_name] = value
        return dict(assays)
    full = fetch(f"{FTP_BASE}/{acc}/{acc}.sdrf.txt",
                 cache / f"{acc}.sdrf.txt", offline)
    if full is None:
        return None
    log(f"{acc}: no condensed SDRF -> using full sdrf.txt fallback")
    return parse_full_sdrf(full)


def assay_annotations(acc, sdrf, report):
    """Reduce SDRF attrs -> {assay_id: dict(donor_id, ethnicity_verbatim,
    disease, sex, tissue)} for assays passing organism check."""
    ann = {}
    donor_fallbacks = 0
    for assay, attrs in sdrf.items():
        org = attrs.get("organism", "")
        if org and "homo sapiens" not in org.lower():
            continue
        eth = None
        for name, value in attrs.items():
            if ETHNICITY_ATTR_RE.search(name):
                eth = value
                break
        donor = next((attrs[a] for a in DONOR_ATTRS if attrs.get(a)), None)
        if donor is None:
            donor = assay
            donor_fallbacks += 1
        disease = next((attrs[a] for a in DISEASE_ATTRS if a in attrs), None)
        ann[assay] = {
            "donor_id": donor,
            "ethnicity_verbatim": "" if eth is None else eth,
            "disease": disease,
            "clinical_information": attrs.get("clinical information", ""),
            "sex": attrs.get("sex", ""),
            "tissue": attrs.get("organism part", ""),
        }
    if donor_fallbacks:
        report["donor_id_fallback_assays"] = donor_fallbacks
        log(f"{acc}: {donor_fallbacks} assays lack a donor attr -> assay id used as donor")
    return ann


def resolve_label(raw, acc, report):
    """Map a verbatim label. Returns (group, subgroup_uncertain) or None."""
    key = normalize_label(raw)
    hit = LABEL_MAP.get(key)
    if hit is None:
        report["unmapped_NEED_REVIEW"][f"{acc} :: {raw}"] += 1
        return None
    if hit == EXCLUDE:
        report["excluded_labels"][f"{acc} :: {raw}"] += 1
        return None
    if hit == UNLABELED:
        report["unlabeled"][f"{acc} :: {raw or '<empty>'}"] += 1
        return None
    return hit


# ---------------------------------------------------------------------------
# Matrix


def load_matrix(acc, cache, offline=False):
    """Fetch + load the raw aggregated counts. Returns (X_csc, genes, cells)
    with X genes x cells, or None if any piece is unavailable."""
    import scipy.io
    import scipy.sparse

    mtx = fetch_first(acc, [".aggregated_counts.mtx.gz", ".aggregated_counts.mtx"],
                      cache, offline)
    rows = fetch_first(acc, [".aggregated_counts.mtx_rows.gz", ".aggregated_counts.mtx_rows"],
                       cache, offline)
    cols = fetch_first(acc, [".aggregated_counts.mtx_cols.gz", ".aggregated_counts.mtx_cols"],
                       cache, offline)
    if not (mtx and rows and cols):
        return None
    with open_maybe_gz(rows) as f:
        genes = [line.rstrip("\n").split("\t")[0] for line in f if line.strip()]
    with open_maybe_gz(cols) as f:
        cells = [line.rstrip("\n").split("\t")[0] for line in f if line.strip()]
    X = scipy.sparse.csc_matrix(scipy.io.mmread(str(mtx)))  # mmread handles .gz
    if X.shape != (len(genes), len(cells)):
        raise ValueError(f"{acc}: matrix {X.shape} != genes x cells "
                         f"({len(genes)}, {len(cells)})")
    return X, genes, cells


def match_cells(cell_ids, ann):
    """Match matrix cell ids to SDRF assay ids (exact, then droplet
    '<assay>-<barcode>' prefix). Returns list of (col_idx, cell_id, assay_id)."""
    out = []
    for i, cid in enumerate(cell_ids):
        if cid in ann:
            out.append((i, cid, cid))
            continue
        stem = re.sub(r"-\d+$", "", cid)  # 10x '-1' style numeric suffix
        if stem in ann:
            out.append((i, cid, stem))
            continue
        if "-" in stem:
            prefix, suffix = stem.rsplit("-", 1)
            if BARCODE_RE.match(suffix) and prefix in ann:
                out.append((i, cid, prefix))
    return out


# ---------------------------------------------------------------------------
# Per-experiment pipeline


def process_experiment(acc, args, report):
    row = {"accession": acc, "status": "ok", "n_assays_sdrf": 0,
           "n_matrix_cells": "", "n_matched": "", "n_kept": 0,
           "n_dropped_unlabeled": 0, "n_dropped_excluded": 0,
           "n_dropped_disease": 0, "n_donors": 0, "n_parts": 0,
           "ensembl_frac": "", "groups": ""}
    exp_report = report["per_experiment"][acc] = {}

    sdrf = load_sdrf(acc, args.sdrf_dir, args.cache / "sdrf", args.offline)
    if sdrf is None:
        row["status"] = "ERROR_no_sdrf"
        log(f"{acc}: condensed SDRF unavailable")
        return row
    ann = assay_annotations(acc, sdrf, exp_report)
    row["n_assays_sdrf"] = len(ann)

    # per-assay label + disease decisions (matrix-independent, so dry-run
    # accounting matches the real run on the SDRF side)
    kept = {}
    for assay, a in ann.items():
        resolved = resolve_label(a["ethnicity_verbatim"], acc, report)
        if resolved is None:
            if normalize_label(a["ethnicity_verbatim"]) in LABEL_MAP:
                key = normalize_label(a["ethnicity_verbatim"])
                if LABEL_MAP.get(key) == EXCLUDE:
                    row["n_dropped_excluded"] += 1
                else:
                    row["n_dropped_unlabeled"] += 1
            else:
                row["n_dropped_excluded"] += 1  # unmapped -> excluded + NEED_REVIEW
            continue
        disease = a["disease"]
        if disease is None and not args.keep_unknown_disease:
            row["n_dropped_disease"] += 1
            exp_report["dropped_disease_missing"] = exp_report.get(
                "dropped_disease_missing", 0) + 1
            # surface what 'clinical information' says about these assays so a
            # curator can decide whether --keep-unknown-disease is safe here
            # (e.g. E-ENAD-21 'reduction mammoplasty' = healthy breast)
            exp_report.setdefault("disease_missing_clinical_info", Counter())[
                a["clinical_information"] or "<none>"] += 1
            continue
        if disease is not None and normalize_label(disease) not in NORMAL_DISEASE:
            row["n_dropped_disease"] += 1
            exp_report.setdefault("dropped_disease_values", Counter())[disease] += 1
            continue
        group, uncertain = resolved
        kept[assay] = dict(a, group=group, subgroup_uncertain=uncertain,
                           disease=disease if disease is not None else "unknown")
    row["n_kept"] = len(kept)
    row["n_donors"] = len({a["donor_id"] for a in kept.values()})
    gcounts = Counter(a["group"] for a in kept.values())
    row["groups"] = json.dumps(dict(sorted(gcounts.items())))
    if not kept:
        row["status"] = "no_usable_assays"
        return row
    if args.dry_run:
        row["status"] = "dry_run"
        return row
    if args.fetch_only:
        # network stage for the login node: populate the cache, no parsing.
        # The compute-node run then passes --offline and touches no network.
        ok = all(fetch_first(acc, sufs, args.cache / acc, args.offline) is not None
                 for sufs in ([".aggregated_counts.mtx.gz", ".aggregated_counts.mtx"],
                              [".aggregated_counts.mtx_rows.gz", ".aggregated_counts.mtx_rows"],
                              [".aggregated_counts.mtx_cols.gz", ".aggregated_counts.mtx_cols"]))
        row["status"] = "fetched" if ok else "ERROR_fetch_incomplete"
        if not ok:
            log(f"{acc}: matrix files incomplete after fetch")
        return row

    loaded = load_matrix(acc, args.cache / acc, args.offline)
    if loaded is None:
        row["status"] = "ERROR_no_matrix"
        log(f"{acc}: raw counts matrix unavailable")
        return row
    X, genes, cells = loaded
    row["n_matrix_cells"] = len(cells)

    matched = match_cells(cells, ann)
    row["n_matched"] = len(matched)
    if len(matched) < 0.5 * len(cells):
        log(f"{acc}: WARNING only {len(matched)}/{len(cells)} matrix cells "
            f"matched SDRF assays")
    triples = [(i, cid, aid) for i, cid, aid in matched if aid in kept]
    if not triples:
        row["status"] = "no_cells_after_join"
        return row

    genes_clean = [VERSION_RE.sub("", g) for g in genes]
    ens_frac = sum(bool(ENSG_RE.match(g)) for g in genes_clean) / max(len(genes_clean), 1)
    row["ensembl_frac"] = round(ens_frac, 4)
    if ens_frac < 0.9:
        log(f"{acc}: WARNING only {ens_frac:.1%} of gene ids look Ensembl (ENSG)")
    n_dup = len(genes_clean) - len(set(genes_clean))
    if n_dup:
        exp_report["duplicate_gene_ids_after_version_strip"] = n_dup

    row["n_parts"] = write_parts(acc, X, genes, genes_clean, triples, kept, args)
    # n_kept now means matrix cells written, not SDRF assays kept
    row["n_kept"] = len(triples)
    row["n_donors"] = len({kept[a]["donor_id"] for _, _, a in triples})
    row["groups"] = json.dumps(dict(sorted(
        Counter(kept[a]["group"] for _, _, a in triples).items())))
    return row


def write_parts(acc, X, genes, genes_clean, triples, kept, args):
    """Slice matched cells into <= chunk-sized cells x genes h5ad parts."""
    import anndata
    import pandas as pd
    import scipy.sparse

    xdir = args.out / "h5ad_parts"
    xdir.mkdir(parents=True, exist_ok=True)
    var = pd.DataFrame({"gene_id_original": genes}, index=pd.Index(genes_clean, name="gene_id"))
    n_parts = int(np.ceil(len(triples) / args.chunk))
    for p in range(n_parts):
        fp = xdir / f"{acc}.part_{p:04d}.h5ad"
        if fp.exists():
            log(f"{acc}: {fp.name} exists, skipping")
            continue
        sl = triples[p * args.chunk:(p + 1) * args.chunk]
        cols = [i for i, _, _ in sl]
        obs = pd.DataFrame({
            "dataset_id": acc,
            "assay_id": [aid for _, _, aid in sl],
            "donor_id": [kept[aid]["donor_id"] for _, _, aid in sl],
            "donor_key": [f"scea|{acc}|{kept[aid]['donor_id']}" for _, _, aid in sl],
            "self_reported_ethnicity": [kept[aid]["ethnicity_verbatim"] for _, _, aid in sl],
            "group": [kept[aid]["group"] for _, _, aid in sl],
            "subgroup_uncertain": [kept[aid]["subgroup_uncertain"] for _, _, aid in sl],
            "provenance": PROVENANCE,
            "disease": [kept[aid]["disease"] for _, _, aid in sl],
            "sex": [kept[aid]["sex"] for _, _, aid in sl],
            "tissue": [kept[aid]["tissue"] for _, _, aid in sl],
        }, index=pd.Index([cid for _, cid, _ in sl], name="cell_id"))
        Xp = scipy.sparse.csr_matrix(X[:, cols].T, dtype=np.float32)
        ad = anndata.AnnData(X=Xp, obs=obs, var=var)
        ad.uns["source"] = "EBI-SCEA"
        ad.uns["source_url"] = f"{FTP_BASE}/{acc}/"
        ad.uns["counts_layer"] = "raw_aggregated_counts"
        tmp = fp.with_suffix(".h5ad.tmp")
        ad.write_h5ad(tmp)
        tmp.rename(fp)
        log(f"{acc}: wrote {fp.name} ({len(sl):,} cells)")
    return n_parts


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="output dir (h5ad_parts/, MANIFEST_scea.csv, report)")
    ap.add_argument("--cache", type=pathlib.Path, default=None,
                    help="download cache (default: <out>/cache)")
    ap.add_argument("--sdrf-dir", type=pathlib.Path, default=None,
                    help="dir with pre-downloaded <EXP>.condensed-sdrf.tsv")
    ap.add_argument("--experiments", type=str, default=None,
                    help="comma-separated accession subset (default: all 18)")
    ap.add_argument("--chunk", type=int, default=100_000,
                    help="max cells per h5ad part (default 100000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="SDRF accounting only; no matrix downloads, no h5ads")
    ap.add_argument("--fetch-only", action="store_true",
                    help="download SDRFs + matrix files into --cache and stop "
                         "(network stage; run on a node with EBI egress, then "
                         "process on a compute node with --offline)")
    ap.add_argument("--keep-unknown-disease", action="store_true",
                    help="keep assays whose SDRF has no disease attribute "
                         "(default: drop them; healthy-only policy)")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; fail on cache misses")
    args = ap.parse_args(argv)

    args.cache = args.cache or args.out / "cache"
    accs = args.experiments.split(",") if args.experiments else EXPERIMENTS
    unknown = [a for a in accs if a not in EXPERIMENTS]
    if unknown:
        ap.error(f"not in the 18-experiment inventory: {unknown}")

    report = {"adapter": "scea_v1", "dry_run": args.dry_run,
              "excluded_labels": Counter(), "unlabeled": Counter(),
              "unmapped_NEED_REVIEW": Counter(), "per_experiment": {}}
    rows = []
    for acc in accs:
        log(f"--- {acc} ---")
        try:
            rows.append(process_experiment(acc, args, report))
        except Exception as e:  # noqa: BLE001 - keep going, record failure
            log(f"{acc}: FAILED: {e!r}")
            rows.append({"accession": acc, "status": f"ERROR_{type(e).__name__}"})

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["accession", "status", "n_assays_sdrf", "n_matrix_cells",
              "n_matched", "n_kept", "n_dropped_unlabeled",
              "n_dropped_excluded", "n_dropped_disease", "n_donors",
              "n_parts", "ensembl_frac", "groups"]
    with open(args.out / "MANIFEST_scea.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for k in ("excluded_labels", "unlabeled", "unmapped_NEED_REVIEW"):
        report[k] = dict(sorted(report[k].items()))
    for exp in report["per_experiment"].values():
        for k in ("dropped_disease_values", "disease_missing_clinical_info"):
            if k in exp:
                exp[k] = dict(exp[k])
    with open(args.out / "scea_run_report.json", "w") as f:
        json.dump(report, f, indent=1)

    log(f"manifest -> {args.out / 'MANIFEST_scea.csv'}")
    log(f"report   -> {args.out / 'scea_run_report.json'}")
    if report["unmapped_NEED_REVIEW"]:
        log(f"NEED_REVIEW unmapped labels: {report['unmapped_NEED_REVIEW']}")
    n_bad = sum(r["status"].startswith("ERROR") for r in rows)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
