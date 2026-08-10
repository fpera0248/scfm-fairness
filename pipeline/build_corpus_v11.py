#!/usr/bin/env python
"""Build tokenized_v2_1: Census tokenized_v2 (reused as-is) + adapter cells.

tokenized_v2 (3,949,268 Census cells) is NEVER re-tokenized: it is loaded
with datasets.load_from_disk and concatenated with newly tokenized DISCO /
SCEA adapter cells. Census cells keep their split / arm_* values untouched;
adapter cells get fresh donor-level splits and incremental arm flags, so
finetune_geneformer.py (which filters split=='train' and arm_*==1) works
unchanged on the new directory.

Adapter policy (v1.1):
  keep          only cells with a non-empty cell_type (SCEA parts need
                adapters/scea_celltype_rescue.py first; dropped counts are
                logged) that MAPS in the coarse cell-type map (--coarse-map,
                the same csv finetune_geneformer.py maps through; unmapped
                values are dropped at selection time and counted PER VALUE
                in v11_report.json -- candidates for map extension) and
                group in the 5 corpus groups (African, EastSEAsian,
                European, HispanicLatino, SouthAsian).
  cap           10k cells/donor (seeded subsample), matching build_corpus.py.
  splits        NEW adapter donors only: per adapter x group, ~10% of donors
                -> split='eval_donor', rest 'train' (donor-level, seeded).
                At least one train donor is always kept, so tiny groups
                deviate from the strict 10%.
  balanced      A = min over groups of pooled adapter TRAIN cells; every
                group contributes exactly A adapter cells with
                arm_balanced=1 (seeded subsample).
  proportional  ALL adapter train cells get arm_proportional=1.
  matched       adapter cells get arm_matched=0 ALWAYS: the tissue-matched
                arm was sampled on Census tissue_general strata, and adapter
                tissue labels are free-text and unharmonized, so the matched
                arm stays Census-only by design.

Census overlap:
  No code-level overlap guard runs here BY DESIGN: the adapter projects
  were pre-deduplicated against Census upstream. The DISCO and SCEA source
  projects are registry rows with DEDUP_STATUS=net-new
  (registry/ethnicity_registry.csv), and Census/SCP duplicates are
  hard-excluded from adapter_disco.NET_NEW_PROJECTS (see adapter_disco.py).
  The dataset_ids of the adapter cells that enter the corpus are written to
  v11_report.json (adapters.<name>.dataset_ids) for manual audit against
  the Census manifest.

Tokenization mirrors tokenize_corpus.py exactly (ensembl_id from the var
index -- adapter parts index var by versionless ENSG --, n_counts, no
filter_pass -> tokenize all cells, V2 dictionaries, same 12 custom attrs).
joinid for adapter cells is the string cell_id; if tokenized_v2 stores
joinid as int64, the CENSUS joinid column is cast to string (schema-only
change, values preserved verbatim) so the concatenate succeeds. Every other
adapter column is cast TO tokenized_v2's features before concatenation.

Resumable: per-part tokenized datasets land under --stage and are skipped
only when both <prefix>.dataset and its <prefix>.done sentinel exist (a
dataset without the sentinel is partial and gets re-tokenized). The stage
is bound to ONE selection: a sha256 fingerprint over the sorted selection
tuples (prefix, row, split, arm flags) plus the args that shaped selection
is written to <stage>/SELECTION_FINGERPRINT; a later run that computes a
different fingerprint (or finds staged parts with no fingerprint) ABORTS
and tells you to use a fresh --stage -- nothing is auto-deleted. After
concatenation, the adapter rows actually in the output are audited against
the selection df per (group, split, arm_balanced, arm_proportional); any
mismatch raises with a diff table.

Outputs:
  --out (default tokenized_v2_1)   concatenated HF dataset
  v11_report.json (beside --out)   per-adapter per-group accounting

Usage:
  python build_corpus_v11.py --adapters disco
  python build_corpus_v11.py --adapters disco,scea
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd

CORPUS = pathlib.Path("/oscar/scratch/fperalta/bibm_corpus")
DEFAULT_PARTS = {"disco": CORPUS / "disco" / "h5ad_parts",
                 "scea": CORPUS / "scea" / "h5ad_parts"}
GROUPS = ["African", "EastSEAsian", "European", "HispanicLatino", "SouthAsian"]
ATTRS = ["joinid", "cell_type", "group", "split", "arm_balanced", "arm_matched",
         "arm_proportional", "donor_key", "dataset_id", "tissue_general",
         "self_reported_ethnicity", "assay"]
EMPTY_CT = {"", "nan", "none", "na", "n/a"}
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS") or 8)


def log(msg):
    print(f"[build_v11] {msg}", flush=True)


def list_parts(adapter, root):
    pats = {"disco": "*/*.h5ad", "scea": "*.h5ad"}
    parts = sorted(root.glob(pats[adapter]))
    if not parts:
        raise SystemExit(f"[build_v11] no {adapter} parts under {root}")
    return parts


PREFIX_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def part_prefix(adapter, root, p):
    """Staging prefix, sanitized to [A-Za-z0-9_-] (dots -> '_').

    Dots are FATAL unsanitized: geneformer's tokenize_data resolves its
    output as (out_dir / prefix).with_suffix('.dataset'), so a dotted
    prefix like 'scea__E-ANND-1.part_0000' would collapse every part of an
    experiment onto one 'scea__E-ANND-1.dataset', silently overwriting.
    Uniqueness of the sanitized prefixes is enforced in scan_adapter."""
    rel = p.relative_to(root).with_suffix("")
    raw = f"{adapter}__" + str(rel).replace("/", "__")
    return PREFIX_UNSAFE_RE.sub("_", raw)


def load_coarse_types(path):
    """Set of mappable cell_type keys from the coarse map csv
    (finetune_geneformer.py drops anything not in this set as 'DROP')."""
    if not path.exists():
        raise SystemExit(f"[build_v11] coarse map not found: {path} "
                         f"(pass --coarse-map)")
    m = pd.read_csv(path)
    if "cell_type" not in m.columns or "coarse_label" not in m.columns:
        raise SystemExit(f"[build_v11] {path} lacks cell_type/coarse_label "
                         f"columns")
    types = set(m["cell_type"].astype(str))
    log(f"coarse map {path}: {len(types)} mappable cell_type values")
    return types


def default_coarse_map():
    here = pathlib.Path(__file__).resolve().parent
    for cand in (here / "celltype_coarse_map.csv",
                 here / "adapters" / "celltype_coarse_map.csv"):
        if cand.exists():
            return cand
    return here / "celltype_coarse_map.csv"  # error surfaces in load_coarse_types


# ---------------------------------------------------------------------------
# Selection (metadata pass; obs only, X never loaded)


def scan_adapter(adapter, root, rep, coarse_types):
    """Scan the adapter's parts -> DataFrame of kept cells (cell_type present
    AND coarse-mappable AND group in the 5 corpus groups) with part/row
    provenance. Unmapped cell_type values are counted per value into the
    report (candidates for coarse-map extension)."""
    import anndata as ad

    parts = list_parts(adapter, root)
    prefixes = [part_prefix(adapter, root, p) for p in parts]
    dupes = sorted(pf for pf, c in Counter(prefixes).items() if c > 1)
    if dupes:
        raise SystemExit(f"[build_v11] sanitized part prefixes collide for "
                         f"{adapter}: {dupes[:5]}{'...' if len(dupes) > 5 else ''} "
                         f"-- rename the offending parts")
    frames = []
    n_scanned = n_no_ct = n_unmapped = parts_missing_ct = 0
    dropped_group = Counter()
    unmapped_ct = Counter()
    dataset_ids = set()
    for p, pfx in zip(parts, prefixes):
        a = ad.read_h5ad(p, backed="r")
        obs = a.obs
        n_scanned += len(obs)
        if "cell_type" in obs.columns:
            ct = obs["cell_type"].astype(str).str.strip()
        else:
            parts_missing_ct += 1
            ct = pd.Series("", index=obs.index)
        has_ct = ~ct.str.lower().isin(EMPTY_CT)
        grp = obs["group"].astype(str)
        good_grp = grp.isin(GROUPS)
        n_no_ct += int((~has_ct).sum())
        for g, c in grp[has_ct & ~good_grp].value_counts().items():
            if c:
                dropped_group[str(g)] += int(c)
        mapped = ct.isin(coarse_types)
        for v, c in ct[has_ct & good_grp & ~mapped].value_counts().items():
            if c:
                unmapped_ct[str(v)] += int(c)
        n_unmapped += int((has_ct & good_grp & ~mapped).sum())
        keep = (has_ct & good_grp & mapped).to_numpy()
        if keep.any():
            if "dataset_id" in obs.columns:
                dataset_ids.update(
                    np.unique(obs["dataset_id"].astype(str).to_numpy()[keep])
                    .tolist())
            frames.append(pd.DataFrame({
                "adapter": adapter,
                "part": str(p),
                "prefix": pfx,
                "row": np.flatnonzero(keep),
                "cell_id": np.asarray(obs.index.astype(str))[keep],
                "cell_type": ct.to_numpy()[keep],
                "group": grp.to_numpy()[keep],
                "donor_key": obs["donor_key"].astype(str).to_numpy()[keep],
            }))
        try:
            a.file.close()
        except Exception:  # noqa: BLE001 - non-backed / older anndata
            pass
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=["adapter", "part", "prefix", "row",
                                     "cell_id", "cell_type", "group",
                                     "donor_key"]))
    if parts_missing_ct:
        log(f"{adapter}: {parts_missing_ct}/{len(parts)} parts have NO "
            f"cell_type column (SCEA parts need scea_celltype_rescue.py first)")
    log(f"{adapter}: {len(parts)} parts, {n_scanned:,} cells scanned -> "
        f"{len(df):,} kept ({n_no_ct:,} dropped without cell_type, "
        f"{sum(dropped_group.values()):,} dropped outside the 5 groups, "
        f"{n_unmapped:,} dropped with cell_type unmapped in the coarse map)")
    if unmapped_ct:
        top = unmapped_ct.most_common(10)
        log(f"{adapter}: top unmapped cell_type values (coarse-map extension "
            f"candidates): {top}")
    rep["adapters"][adapter] = {
        "n_parts": len(parts),
        "cells_scanned": n_scanned,
        "dropped_no_celltype": n_no_ct,
        "dropped_out_of_scope_group": dict(sorted(dropped_group.items())),
        "dropped_unmapped_celltype": n_unmapped,
        "unmapped_celltype_values": dict(
            sorted(unmapped_ct.items(), key=lambda kv: (-kv[1], kv[0]))),
        "parts_missing_celltype_column": parts_missing_ct,
        "dataset_ids": sorted(dataset_ids),
    }
    return df


def apply_donor_cap(df, cap, rng, rep):
    keep, n_capped, n_dropped = [], 0, 0
    for dk, idx in df.groupby("donor_key", sort=True).indices.items():
        if len(idx) > cap:
            keep.append(rng.choice(idx, size=cap, replace=False))
            n_capped += 1
            n_dropped += len(idx) - cap
            log(f"donor cap: {dk} {len(idx):,} -> {cap:,}")
        else:
            keep.append(idx)
    df = df.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    log(f"donor cap {cap:,}: {n_capped} donors capped, {n_dropped:,} cells "
        f"dropped -> {len(df):,} cells")
    rep["donor_cap"] = {"cap": cap, "donors_capped": n_capped,
                       "cells_dropped": n_dropped}
    return df


def assign_splits(df, frac, rng):
    """Donor-level holdout per adapter x group: ~frac of donors ->
    'eval_donor', rest 'train'. Always keeps >=1 train donor per group."""
    df["split"] = "train"
    for (adapter, g), sub in df.groupby(["adapter", "group"], sort=True):
        donors = np.sort(sub["donor_key"].unique())
        n = len(donors)
        n_hold = 0 if n <= 1 else min(n - 1, max(1, int(round(n * frac))))
        if n_hold:
            hold = set(rng.choice(donors, size=n_hold, replace=False))
            df.loc[sub.index[sub["donor_key"].isin(hold)], "split"] = "eval_donor"
        log(f"splits {adapter}/{g}: {n} donors -> {n_hold} eval_donor")
    return df


def assign_arms(df, rng, rep):
    """Incremental arm flags for adapter cells only (census untouched):
    balanced = A per group, proportional = all train, matched = 0 always."""
    for arm in ("arm_balanced", "arm_matched", "arm_proportional"):
        df[arm] = 0
    train = df[df["split"] == "train"]
    counts = train.groupby("group").size().reindex(GROUPS, fill_value=0)
    A = int(counts.min())
    log(f"adapter train cells/group: "
        f"{ {g: int(counts[g]) for g in GROUPS} } -> balanced A={A:,}")
    if A == 0:
        log("WARNING: at least one group has 0 adapter train cells -> the "
            "balanced arm gains nothing from the adapters")
    for g in GROUPS:
        pool = train.index[train["group"] == g].to_numpy()
        if A and len(pool) >= A:
            df.loc[rng.choice(pool, size=A, replace=False), "arm_balanced"] = 1
    df.loc[train.index, "arm_proportional"] = 1
    rep["balanced_A_per_group"] = A
    rep["adapter_train_cells_per_group"] = {g: int(counts[g]) for g in GROUPS}
    return df


def per_group_accounting(df, rep):
    for adapter, adf in df.groupby("adapter", sort=True):
        pg = {}
        for g in GROUPS:
            gdf = adf[adf["group"] == g]
            tr = gdf[gdf["split"] == "train"]
            ev = gdf[gdf["split"] == "eval_donor"]
            pg[g] = {
                "train_cells": len(tr),
                "eval_donor_cells": len(ev),
                "train_donors": int(tr["donor_key"].nunique()),
                "eval_donors": int(ev["donor_key"].nunique()),
                "arm_balanced_added": int((gdf["arm_balanced"] == 1).sum()),
                "arm_proportional_added": int((gdf["arm_proportional"] == 1).sum()),
                "arm_matched_added": 0,
            }
        rep["adapters"][adapter]["per_group"] = pg
        rep["adapters"][adapter]["cells_after_cap"] = len(adf)
        rep["adapters"][adapter]["donors"] = int(adf["donor_key"].nunique())


# ---------------------------------------------------------------------------
# Stage fingerprint (staged parts bake split/arm flags; the stage is only
# valid for the exact selection that created it)


def selection_fingerprint(df, args, adapters):
    """sha256 over the args that shaped selection plus every selected
    (prefix, row, split, arm flags) tuple, sorted."""
    h = hashlib.sha256()
    h.update(json.dumps({"adapters": sorted(adapters), "seed": args.seed,
                         "cap": args.cap, "holdout_frac": args.holdout_frac},
                        sort_keys=True).encode())
    tuples = sorted(zip(df["prefix"].astype(str),
                        df["row"].astype(int),
                        df["split"].astype(str),
                        df["arm_balanced"].astype(int),
                        df["arm_matched"].astype(int),
                        df["arm_proportional"].astype(int)))
    for t in tuples:
        h.update(repr(t).encode())
    return h.hexdigest()


def check_stage_fingerprint(stage, fingerprint):
    """Bind --stage to one selection. ABORT (never auto-delete) when the
    stage was built from a different selection."""
    fp_file = stage / "SELECTION_FINGERPRINT"
    if fp_file.exists():
        recorded = fp_file.read_text().strip()
        if recorded != fingerprint:
            raise SystemExit(
                "[build_v11] STALE STAGE: selection fingerprint mismatch\n"
                f"  stage:    {stage}\n"
                f"  recorded: {recorded}\n"
                f"  current:  {fingerprint}\n"
                "The staged tokenized parts were built from a DIFFERENT "
                "selection\n(other --adapters/--seed/--cap/--holdout-frac/"
                "--coarse-map, or changed\nadapter parts on disk). Their baked "
                "split/arm flags do not match this\nrun's selection, so reusing "
                "them would corrupt the corpus. Use a fresh\n--stage for this "
                "run; this tool never auto-deletes a stage.")
        log("stage fingerprint matches -> resuming staged parts")
        return
    staged = (sorted((stage / "parts").glob("*.dataset"))
              if (stage / "parts").exists() else [])
    if staged:
        raise SystemExit(
            f"[build_v11] STALE STAGE: {len(staged)} staged tokenized parts "
            f"exist under\n  {stage}\nbut no SELECTION_FINGERPRINT records "
            "which selection built them (pre-fingerprint stage?). Use a fresh "
            "--stage; this tool never auto-deletes a stage.")
    stage.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fingerprint + "\n")
    log(f"selection fingerprint -> {fp_file}")


# ---------------------------------------------------------------------------
# Post-concat audit


AUDIT_COLS = ["group", "split", "arm_balanced", "arm_proportional"]


def _audit_counts(frame):
    f = frame[AUDIT_COLS].copy()
    f["group"] = f["group"].astype(str)
    f["split"] = f["split"].astype(str)
    f["arm_balanced"] = f["arm_balanced"].astype(int)
    f["arm_proportional"] = f["arm_proportional"].astype(int)
    return f.groupby(AUDIT_COLS).size().sort_index()


def audit_adapter_rows(add, df):
    """Recompute per-(group, split, arm_balanced, arm_proportional) counts
    from the adapter rows actually in the output and assert they equal the
    selection df's expectation; raise with a diff table on mismatch (this is
    the tripwire for stale staged parts and tokenizer-side cell drops)."""
    actual = add.remove_columns(
        [c for c in add.column_names if c not in AUDIT_COLS]).to_pandas()
    exp = _audit_counts(df)
    act = _audit_counts(actual)
    if len(add) == len(df) and exp.equals(act):
        log(f"post-concat audit passed: {len(add):,} adapter cells match the "
            f"selection df on (group, split, arm_balanced, arm_proportional)")
        return
    diff = (pd.concat([exp.rename("expected"), act.rename("actual")], axis=1)
            .fillna(0).astype(int))
    diff = diff[diff["expected"] != diff["actual"]]
    raise SystemExit(
        "[build_v11] POST-CONCAT AUDIT FAILED: adapter rows in the output do "
        f"not match the selection df ({len(add):,} rows vs {len(df):,} "
        "selected).\nMismatched (group, split, arm_balanced, "
        "arm_proportional) cells:\n"
        f"{diff.to_string()}\n"
        "Likely causes: stale staged parts or tokenizer-side cell drops. "
        "Rebuild with a fresh --stage.")


# ---------------------------------------------------------------------------
# Tokenization (mirrors tokenize_corpus.py)


def make_tokenizer():
    from geneformer import TranscriptomeTokenizer

    tk_kwargs = {}
    cand_dirs = [
        pathlib.Path(os.environ.get("GENEFORMER_ASSETS", "/nonexistent")),
        pathlib.Path.home() / "data/fperalta/Geneformer/geneformer_repo/geneformer",
        pathlib.Path.home() / "data/fperalta/geneformer_repo/geneformer",
    ]
    for d in cand_dirs:
        if (d / "token_dictionary_gc104M.pkl").exists():
            tk_kwargs = {
                "token_dictionary_file": str(d / "token_dictionary_gc104M.pkl"),
                "gene_median_file": str(d / "gene_median_dictionary_gc104M.pkl"),
                "gene_mapping_file": str(d / "ensembl_mapping_dict_gc104M.pkl"),
            }
            log(f"using V2 dictionaries from {d}")
            break
    else:
        log("using geneformer package default dictionaries")
    try:
        return TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                      nproc=NPROC, model_version="V2", **tk_kwargs)
    except TypeError:  # older package without model_version kwarg
        log("NOTE: package lacks model_version kwarg -- verify dictionary "
            "vintage matches V2")
        return TranscriptomeTokenizer(custom_attr_name_dict={a: a for a in ATTRS},
                                      nproc=NPROC, **tk_kwargs)


def tokenize_selection(df, stage):
    """Tokenize the selected adapter cells per part (resumable). Returns the
    list of per-part tokenized dataset paths for the CURRENT selection."""
    import anndata as ad

    tokd = stage / "parts"
    hstage = stage / "h5ad"
    tokd.mkdir(parents=True, exist_ok=True)
    hstage.mkdir(parents=True, exist_ok=True)
    tk = None
    prefixes = sorted(df["prefix"].unique())
    for prefix in prefixes:
        outp = tokd / f"{prefix}.dataset"
        done = tokd / f"{prefix}.done"
        if outp.exists() and done.exists():
            log(f"{prefix}: tokenized part exists, skipping")
            continue
        if outp.exists():
            log(f"{prefix}: tokenized part has NO .done sentinel (killed "
                f"mid-write?) -> re-tokenizing")
            shutil.rmtree(outp)
        sub = df[df["prefix"] == prefix].sort_values("row")
        a = ad.read_h5ad(sub["part"].iloc[0])
        a = a[sub["row"].to_numpy()].copy()
        a.obs["joinid"] = sub["cell_id"].to_numpy()  # string cell_id (see docstring)
        a.obs["cell_type"] = sub["cell_type"].to_numpy()
        a.obs["group"] = sub["group"].to_numpy()
        a.obs["split"] = sub["split"].to_numpy()
        for arm in ("arm_balanced", "arm_matched", "arm_proportional"):
            a.obs[arm] = sub[arm].to_numpy().astype(int)
        a.obs["donor_key"] = sub["donor_key"].to_numpy()
        a.obs["tissue_general"] = (a.obs["tissue"].astype(str)
                                   if "tissue" in a.obs.columns else "unknown")
        if "assay" not in a.obs.columns:
            a.obs["assay"] = "unknown"
        # dataset_id + self_reported_ethnicity already on the part obs
        a.obs["ensembl_ok"] = True
        a.var["ensembl_id"] = a.var_names.to_numpy()  # var index = versionless ENSG
        a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
        for col in ("joinid", "cell_type", "dataset_id", "tissue_general",
                    "self_reported_ethnicity", "assay", "group", "split",
                    "donor_key"):
            a.obs[col] = a.obs[col].astype(str)

        if tk is None:
            tk = make_tokenizer()
        pdir = hstage / prefix
        if pdir.exists():
            shutil.rmtree(pdir)
        pdir.mkdir(parents=True)
        a.write_h5ad(pdir / f"{prefix}.h5ad")
        n = a.n_obs
        del a
        tk.tokenize_data(str(pdir), str(tokd), prefix, file_format="h5ad")
        shutil.rmtree(pdir)
        done.write_text("ok\n")  # resume sentinel: only now is the part complete
        log(f"{prefix}: tokenized ({n:,} cells)")
    return [tokd / f"{p}.dataset" for p in prefixes]


# ---------------------------------------------------------------------------
# Concatenate + save


def concat_and_save(part_paths, df, args, rep):
    from datasets import Features, Value, concatenate_datasets, load_from_disk

    census = load_from_disk(str(args.census_tokenized))
    rep["n_census_cells"] = len(census)

    # census balanced floor (train cells with arm_balanced==1, per group)
    old_floor = None
    try:
        meta = census.remove_columns(
            [c for c in census.column_names
             if c not in ("group", "split", "arm_balanced")]).to_pandas()
        floors = (meta[(meta["split"] == "train") & (meta["arm_balanced"] == 1)]
                  .groupby("group").size())
        old_floor = int(floors.min())
        rep["census_balanced_per_group"] = {str(k): int(v)
                                            for k, v in floors.items()}
    except Exception as e:  # noqa: BLE001 - accounting only, never fatal
        log(f"census floor accounting failed ({e!r}); continuing")
    rep["census_balanced_floor"] = old_floor
    if old_floor is not None:
        rep["new_balanced_floor"] = old_floor + rep["balanced_A_per_group"]

    # dtype alignment: adapter columns are cast TO tokenized_v2's features.
    # The one exception is joinid: adapter joinids are string cell_ids and
    # cannot become int64, so an int64 census joinid is cast to string
    # (schema-only; values preserved verbatim).
    target = Features(dict(census.features))
    rep["joinid_cast_to_string"] = False
    if target["joinid"] != Value("string"):
        log(f"census joinid feature is {target['joinid']} -> casting census "
            f"joinid to string so adapter cell_id joinids concatenate")
        target["joinid"] = Value("string")
        # keep the cast's arrow cache OUT of the pristine tokenized_v2 dir
        # (datasets writes map/cast caches next to a load_from_disk dataset
        # by default): redirect it into the stage dir.
        cast_dir = args.stage / "census_cast"
        cast_dir.mkdir(parents=True, exist_ok=True)
        try:
            census = census.cast(
                target,
                cache_file_name=str(cast_dir / "census_joinid_str.arrow"))
        except TypeError:  # datasets version without cache_file_name on cast
            log("WARNING: this datasets version's cast() lacks "
                "cache_file_name; cast cache files will land beside the "
                "census dataset dir")
            census = census.cast(target)
        rep["joinid_cast_to_string"] = True  # schema-only; values verbatim

    adds = []
    for p in part_paths:
        d = load_from_disk(str(p))
        if d.features != target:
            d = d.cast(target)
        adds.append(d)
    add = concatenate_datasets(adds) if len(adds) > 1 else adds[0]
    if add.column_names != census.column_names and hasattr(add, "select_columns"):
        add = add.select_columns(census.column_names)  # align column order
    rep["n_adapter_cells_added"] = len(add)

    # tripwire: the adapter rows going into the output must match the
    # selection df exactly (stale stage / tokenizer drops fail loudly here)
    audit_adapter_rows(add, df)
    rep["post_concat_audit"] = "passed"

    out = args.out
    if out.resolve() == args.census_tokenized.resolve():
        raise SystemExit("[build_v11] refusing to overwrite the census "
                         "tokenized dir")
    full = concatenate_datasets([census, add])
    # atomic publish: save to <out>.tmp, keep any previous corpus until the
    # new one is fully on disk, then swap.
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    full.save_to_disk(str(tmp))
    prev = None
    if out.exists():
        prev = out.parent / (out.name + ".prev")
        if prev.exists():
            shutil.rmtree(prev)
        out.rename(prev)
    tmp.rename(out)
    if prev is not None:
        shutil.rmtree(prev)
    rep["n_total_cells"] = len(full)
    log(f"BUILD COMPLETE: {len(census):,} census + {len(add):,} adapter "
        f"= {len(full):,} cells -> {out}")


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapters", default="disco",
                    help="comma/space separated subset of: disco scea "
                         "(default: disco)")
    ap.add_argument("--disco-parts", type=pathlib.Path,
                    default=DEFAULT_PARTS["disco"])
    ap.add_argument("--scea-parts", type=pathlib.Path,
                    default=DEFAULT_PARTS["scea"])
    ap.add_argument("--census-tokenized", type=pathlib.Path,
                    default=CORPUS / "tokenized_v2",
                    help="existing tokenized Census dataset (reused as-is)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=CORPUS / "tokenized_v2_1")
    ap.add_argument("--stage", type=pathlib.Path,
                    default=CORPUS / "tok_v11_stage",
                    help="resumable staging dir (per-part tokenized datasets), "
                         "bound to one selection via SELECTION_FINGERPRINT; "
                         "use a fresh dir when the selection changes")
    ap.add_argument("--coarse-map", type=pathlib.Path, default=None,
                    help="coarse cell-type map csv (cell_type,coarse_label); "
                         "adapter cells whose cell_type is not in the map are "
                         "dropped at selection time (default: "
                         "celltype_coarse_map.csv beside this script, or in "
                         "its adapters/ subdir)")
    ap.add_argument("--report", type=pathlib.Path, default=None,
                    help="report path (default: <out parent>/v11_report.json)")
    ap.add_argument("--cap", type=int, default=10_000,
                    help="max cells per donor (default 10000)")
    ap.add_argument("--holdout-frac", type=float, default=0.10,
                    help="fraction of adapter donors per group held out to "
                         "eval_donor (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    adapters = [a for a in args.adapters.replace(",", " ").split() if a]
    unknown = [a for a in adapters if a not in DEFAULT_PARTS]
    if unknown:
        ap.error(f"unknown adapters {unknown}; choose from {sorted(DEFAULT_PARTS)}")
    roots = {"disco": args.disco_parts, "scea": args.scea_parts}

    coarse_map = args.coarse_map or default_coarse_map()
    coarse_types = load_coarse_types(coarse_map)

    rng = np.random.default_rng(args.seed)
    rep = {"builder": "build_corpus_v11", "seed": args.seed, "cap": args.cap,
           "holdout_frac": args.holdout_frac,
           "adapters_included": adapters,
           "coarse_map": str(coarse_map),
           "census_tokenized": str(args.census_tokenized),
           "out": str(args.out), "adapters": {}}

    df = pd.concat([scan_adapter(a, roots[a], rep, coarse_types)
                    for a in adapters],
                   ignore_index=True)
    if df.empty:
        raise SystemExit("[build_v11] no adapter cells survived the "
                         "cell_type+coarse-map+group filter; nothing to add")
    df = apply_donor_cap(df, args.cap, rng, rep)
    df = assign_splits(df, args.holdout_frac, rng)
    df = assign_arms(df, rng, rep)
    per_group_accounting(df, rep)

    fingerprint = selection_fingerprint(df, args, adapters)
    rep["selection_fingerprint"] = fingerprint
    check_stage_fingerprint(args.stage, fingerprint)

    part_paths = tokenize_selection(df, args.stage)
    concat_and_save(part_paths, df, args, rep)

    report_path = args.report or args.out.parent / "v11_report.json"
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=1)
    log(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
