#!/usr/bin/env python
"""HuBMAP v1 adapter for the ethnicity-balanced corpus.

Enumerates public *processed* RNAseq-family HuBMAP datasets whose donor has a
mappable US-census race / Hispanic ethnicity, downloads the processed h5ad per
dataset (resume + sha check), applies standard QC (HuBMAP is unfiltered),
normalizes var to Ensembl gene IDs, harmonizes labels to our 5 macro-groups
(Hispanic ethnicity trumps race; US-census 'Asian' is unsplittable -> group
'EastSEAsian' with subgroup_uncertain=True), and emits per-dataset h5ads shaped
like our Census parts, plus MANIFEST_hubmap.csv.

Standalone: requests + anndata + pandas (numpy/scipy come with anndata).
Dry-run mode enumerates and counts only.
"""
import argparse
import hashlib
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

SEARCH_URL = "https://search.api.hubmapconsortium.org/v3/portal/search"
ASSETS_URL = "https://assets.hubmapconsortium.org"

RNA_FAMILY = ["RNAseq", "SNARE-seq2", "10X Multiome", "Visium (no probes)", "Slide-seq",
              "GeoMx (NGS)", "Xenium", "seqFISH", "CosMx Transcriptomics", "MUSIC"]

# preference order for the expression file; expr.h5ad is the unfiltered
# counts matrix from the salmon pipeline, which is what we QC ourselves
H5AD_PREFERENCE = ["expr.h5ad", "raw_expr.h5ad", "secondary_analysis.h5ad"]

CELLTYPE_COLS = ["predicted.ASCT.celltype", "azimuth_label", "predicted_label",
                 "predicted_celltype", "cell_type"]

PROVENANCE = "hubmap_us_census_race"
MIN_GENES = 200
MAX_MITO_FRAC = 0.20
JOINID_BLOCK = 10_000_000  # per-dataset joinid block; negative ids, no Census collision

# US-census race -> (macro group, subgroup_uncertain); Hispanic ethnicity trumps race
RACE_TO_GROUP = {
    "White": ("European", False),
    "Black or African American": ("African", False),
    "Asian": ("EastSEAsian", True),  # census 'Asian' cannot split South vs E/SE
}
HISPANIC = "Hispanic or Latino"

# mapped_organ (laterality stripped, lowercased) -> Census tissue_general
ORGAN_OVERRIDES = {
    "large intestine": "large intestine",
    "small intestine": "small intestine",
    "muscle": "musculature",
    "blood vasculature": "vasculature",
    "aorta": "vasculature",
    "mammary gland": "breast",
    "bladder": "bladder organ",
}

# 13 mito protein-coding genes; enough for a mito fraction on Ensembl vars
MITO_ENSG = {
    "ENSG00000198888", "ENSG00000198763", "ENSG00000198804", "ENSG00000198712",
    "ENSG00000228253", "ENSG00000198899", "ENSG00000198938", "ENSG00000198840",
    "ENSG00000212907", "ENSG00000198886", "ENSG00000198786", "ENSG00000198695",
    "ENSG00000198727"}


def first(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x


def search_datasets(sess):
    query = {
        "size": 10000,
        "query": {"bool": {
            "must": [
                {"term": {"entity_type.keyword": "Dataset"}},
                {"term": {"processing.keyword": "processed"}},
                {"term": {"mapped_data_access_level.keyword": "Public"}},
                {"terms": {"raw_dataset_type.keyword": RNA_FAMILY}},
            ],
            "must_not": [{"term": {"is_component": True}}],
        }},
        "_source": ["uuid", "hubmap_id", "raw_dataset_type", "assay_display_name",
                    "mapped_data_access_level", "mapped_status", "files",
                    "registered_doi", "doi_url",
                    "origin_samples.mapped_organ",
                    "donor.uuid", "donor.hubmap_id",
                    "donor.mapped_metadata.race", "donor.mapped_metadata.ethnicity",
                    "donor.mapped_metadata.age_value", "donor.mapped_metadata.age_unit",
                    "donor.mapped_metadata.sex", "donor.mapped_metadata.medical_history"],
    }
    for attempt in range(3):
        try:
            r = sess.post(SEARCH_URL, json=query, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"search retry after error: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    d = r.json()
    total = d["hits"]["total"]["value"]
    hits = d["hits"]["hits"]
    if total > len(hits):
        print(f"WARNING: search window returned {len(hits)}/{total} hits",
              file=sys.stderr)
    return [h["_source"] for h in hits]


def map_group(race, ethnicity):
    """Returns (group, subgroup_uncertain, original_label) or (None, ...)"""
    if ethnicity == HISPANIC:
        return "HispanicLatino", False, ethnicity
    hit = RACE_TO_GROUP.get(race or "")
    if hit:
        return hit[0], hit[1], race
    return None, False, race


def tissue_general(mapped_organ):
    if not mapped_organ:
        return "unknown"
    organ = re.sub(r"\s*\((Left|Right)\)\s*$", "", mapped_organ).strip().lower()
    return ORGAN_OVERRIDES.get(organ, organ)


def pick_h5ad(files):
    files = files or []
    by_base = {pathlib.PurePosixPath(f.get("rel_path", "")).name: f for f in files
               if f.get("rel_path", "").endswith(".h5ad")}
    for name in H5AD_PREFERENCE:
        if name in by_base:
            return by_base[name]
    return None


def enumerate_datasets(sess):
    rows = []
    for src in search_datasets(sess):
        donor = src.get("donor") or {}
        mm = donor.get("mapped_metadata") or {}
        race, eth = first(mm.get("race")), first(mm.get("ethnicity"))
        group, uncertain, orig = map_group(race, eth)
        if group is None:
            continue
        age_v, age_u = first(mm.get("age_value")), first(mm.get("age_unit"))
        history = mm.get("medical_history") or []
        f = pick_h5ad(src.get("files"))
        organs = src.get("origin_samples") or []
        organs = organs if isinstance(organs, list) else [organs]
        organ = first([o.get("mapped_organ") for o in organs if o.get("mapped_organ")])
        rows.append({
            "dataset_uuid": src["uuid"],
            "hubmap_id": src.get("hubmap_id"),
            "donor_uuid": donor.get("uuid"),
            "donor_id": donor.get("hubmap_id"),
            "race": race, "ethnicity": eth,
            "self_reported_ethnicity": orig,
            "group": group, "subgroup_uncertain": uncertain,
            "age": f"{age_v:g} {age_u}" if age_v is not None and age_u else age_v,
            "sex": first(mm.get("sex")),
            "disease": "; ".join(history) if history else "unknown",
            "assay": first(src.get("assay_display_name")),
            "raw_dataset_type": src.get("raw_dataset_type"),
            "organ": organ,
            "tissue_general": tissue_general(organ),
            "doi": src.get("registered_doi") or src.get("doi_url"),
            "h5ad_rel_path": f.get("rel_path") if f else None,
            "h5ad_size": f.get("size") if f else None,
            "h5ad_expected_sha256": (f or {}).get("sha256") or (f or {}).get("checksum"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("dataset_uuid").reset_index(drop=True)
    return df


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_asset(sess, dataset_uuid, rel_path, dest, expected_sha=None):
    """Resumable download from assets.hubmapconsortium.org; returns sha256."""
    url = f"{ASSETS_URL}/{dataset_uuid}/{rel_path}"
    if dest.exists():
        got = sha256_file(dest)
        if expected_sha is None or got == expected_sha:
            return got
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers, mode = {}, "wb"
    if tmp.exists() and tmp.stat().st_size > 0:
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    with sess.get(url, headers=headers, stream=True, timeout=300) as r:
        if r.status_code == 416:
            pass  # .part already complete
        elif r.status_code in (200, 206):
            if r.status_code == 200:
                mode = "wb"  # server ignored Range; restart
            with open(tmp, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        else:
            r.raise_for_status()
    got = sha256_file(tmp)
    if expected_sha is not None and got != expected_sha:
        tmp.unlink()
        raise RuntimeError(f"sha mismatch for {url}: got {got}, want {expected_sha}")
    tmp.rename(dest)
    return got


def load_gene_map(path):
    """TSV symbol -> Ensembl ID (versionless). Header optional."""
    gm = pd.read_csv(path, sep="\t", header=None, dtype=str, comment="#")
    if str(gm.iloc[0, 1]).lower().startswith(("ensembl", "gene")):
        gm = gm.iloc[1:]
    mapping = {}
    for sym, ens in zip(gm.iloc[:, 0], gm.iloc[:, 1]):
        if isinstance(sym, str) and isinstance(ens, str) and ens.startswith("ENSG"):
            mapping[sym.upper()] = ens.split(".")[0]
    return mapping


def to_ensembl(var_names, gene_map):
    """Map var names to versionless Ensembl IDs; None for unmappable."""
    names = [str(v) for v in var_names]
    n_ens = sum(n.startswith("ENSG") for n in names)
    if n_ens >= 0.5 * len(names):
        return [n.split(".")[0] if n.startswith("ENSG") else None for n in names]
    if gene_map is None:
        return None  # symbols but no mapping TSV
    return [gene_map.get(n.upper()) for n in names]


def collapse_by_ensembl(X, ens_ids):
    """Drop unmapped columns, collapse duplicate Ensembl IDs by sum."""
    keep = np.array([e is not None for e in ens_ids])
    X = X[:, keep]
    ids = np.array([e for e in ens_ids if e is not None])
    uniq, inv = np.unique(ids, return_inverse=True)
    if len(uniq) < len(ids):
        M = sp.csr_matrix((np.ones(len(inv), dtype=np.float32),
                           (np.arange(len(inv)), inv)),
                          shape=(len(inv), len(uniq)))
        X = X @ M
    else:
        X = X[:, np.argsort(ids)]
    return sp.csr_matrix(X, dtype=np.float32), uniq


def extract_counts(ad0, filename):
    """Counts matrix + var names + per-cell cell_type series (or None)."""
    if filename == "secondary_analysis.h5ad" and ad0.raw is not None:
        X, var_names = ad0.raw.X, ad0.raw.var_names
    else:
        X, var_names = ad0.X, ad0.var_names
    cell_type = None
    for col in CELLTYPE_COLS:
        if col in ad0.obs.columns:
            cell_type = ad0.obs[col].astype(str).values
            break
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    return sp.csr_matrix(X, dtype=np.float32), var_names, cell_type


def process_dataset(row, dest_h5ad, download_path, gene_map, joinid_base):
    import anndata

    ad0 = anndata.read_h5ad(download_path)
    X, var_names, cell_type = extract_counts(ad0, download_path.name)
    ens = to_ensembl(var_names, gene_map)
    if ens is None:
        raise RuntimeError("var names are symbols; pass --gene-map TSV")
    X, ensembl_ids = collapse_by_ensembl(X, ens)

    # standard QC: HuBMAP processed matrices are unfiltered
    n_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    mito_cols = np.flatnonzero(np.isin(ensembl_ids, list(MITO_ENSG)))
    mito = np.asarray(X[:, mito_cols].sum(axis=1)).ravel() if len(mito_cols) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mito_frac = np.where(n_counts > 0, mito / n_counts, 1.0)
    keep = (n_genes >= MIN_GENES) & (mito_frac < MAX_MITO_FRAC)
    n_total, n_kept = X.shape[0], int(keep.sum())
    if n_kept == 0:
        return n_total, 0
    X = X[keep]

    obs = pd.DataFrame({
        "soma_joinid": joinid_base - np.arange(n_kept, dtype=np.int64),
        "dataset_id": row.dataset_uuid,
        "donor_id": row.donor_id,
        "tissue_general": row.tissue_general,
        "self_reported_ethnicity": row.self_reported_ethnicity,
        "cell_type": cell_type[keep] if cell_type is not None else "unknown",
        "assay": row.assay,
        "disease": row.disease,
        "group": row.group,
        "subgroup_uncertain": bool(row.subgroup_uncertain),
        "donor_key": f"hubmap|{row.donor_uuid}",
        "provenance": PROVENANCE,
        "n_counts": np.asarray(X.sum(axis=1)).ravel(),
    })
    obs.index = obs.soma_joinid.astype(str)
    var = pd.DataFrame({"ensembl_id": ensembl_ids}, index=ensembl_ids)
    var.index.name = "ensembl_id"
    anndata.AnnData(X=X, obs=obs, var=var).write_h5ad(dest_h5ad, compression="gzip")
    return n_total, n_kept


def print_summary(df):
    donors = df.groupby("group").agg(datasets=("dataset_uuid", "size"),
                                     donors=("donor_uuid", "nunique"))
    print(donors.to_string())
    print(f"total: {len(df)} datasets, {df.donor_uuid.nunique()} donors, "
          f"{df.h5ad_rel_path.notna().sum()} with a processed h5ad")


def main():
    ap = argparse.ArgumentParser(description="HuBMAP v1 corpus adapter")
    ap.add_argument("--outdir", type=pathlib.Path,
                    help="output dir for per-dataset h5ads + manifest")
    ap.add_argument("--gene-map", help="TSV symbol->Ensembl for symbol var names")
    ap.add_argument("--dry-run", action="store_true", help="enumerate + count only")
    ap.add_argument("--limit", type=int, help="process at most N datasets")
    args = ap.parse_args()
    if not args.dry_run and args.outdir is None:
        ap.error("--outdir is required unless --dry-run")

    sess = requests.Session()
    sess.headers["User-Agent"] = "scfm-corpus-hubmap-adapter/1.0"
    df = enumerate_datasets(sess)
    if df.empty:
        print("no mappable public processed RNA datasets found", file=sys.stderr)
        sys.exit(1)
    print_summary(df)
    if args.dry_run:
        return

    df = df[df.h5ad_rel_path.notna()].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    gene_map = load_gene_map(args.gene_map) if args.gene_map else None

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    dl_dir = outdir / "downloads"
    dl_dir.mkdir(exist_ok=True)
    manifest_fp = outdir / "MANIFEST_hubmap.csv"
    manifest = []

    for i, row in df.iterrows():
        dest = outdir / f"hubmap_{row.dataset_uuid}.h5ad"
        rec = {"dataset_uuid": row.dataset_uuid, "hubmap_id": row.hubmap_id,
               "donor_uuid": row.donor_uuid, "donor_id": row.donor_id,
               "race": row.race, "ethnicity": row.ethnicity, "age": row.age,
               "sex": row.sex, "group": row.group,
               "subgroup_uncertain": row.subgroup_uncertain, "assay": row.assay,
               "organ": row.organ, "tissue_general": row.tissue_general,
               "doi": row.doi, "h5ad_sha256": None,
               "cells_total": None, "cells_kept": None, "status": None}
        try:
            if dest.exists():
                rec["status"] = "already_done"
            else:
                raw = dl_dir / f"{row.dataset_uuid}_{pathlib.PurePosixPath(row.h5ad_rel_path).name}"
                rec["h5ad_sha256"] = download_asset(
                    sess, row.dataset_uuid, row.h5ad_rel_path, raw,
                    expected_sha=row.h5ad_expected_sha256)
                n_total, n_kept = process_dataset(
                    row, dest, raw, gene_map, joinid_base=-(i * JOINID_BLOCK + 1))
                rec.update(cells_total=n_total, cells_kept=n_kept,
                           status="ok" if n_kept else "empty_after_qc")
        except Exception as e:
            rec["status"] = f"ERROR: {e}"
            print(f"[{i+1}/{len(df)}] {row.dataset_uuid} FAILED: {e}", file=sys.stderr)
        else:
            print(f"[{i+1}/{len(df)}] {row.dataset_uuid} {rec['status']} "
                  f"(kept {rec['cells_kept']})", flush=True)
        manifest.append(rec)
        pd.DataFrame(manifest).to_csv(manifest_fp, index=False)

    done = sum(r["status"] in ("ok", "already_done") for r in manifest)
    print(f"finished: {done}/{len(manifest)} datasets ok -> {manifest_fp}")


if __name__ == "__main__":
    main()
