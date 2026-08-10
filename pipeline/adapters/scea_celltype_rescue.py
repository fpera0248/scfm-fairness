#!/usr/bin/env python
"""Rescue cell types for SCEA h5ad parts from the atlas cell_metadata TSVs.

adapter_scea.py parts carry donor/group/disease/sex/tissue but NO cell_type
column: the condensed SDRF has no per-cell annotation layer. SCEA does
publish a per-cell metadata file on the FTP next to the matrices (verified:
E-ANND-1/E-ANND-1.cell_metadata.tsv; layout
https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/<EXP>/),
which carries inferred / authors cell type columns. For every experiment
with parts under --parts-dir this tool fetches <EXP>.cell_metadata.tsv,
picks the best cell-type column (authors > inferred > any cell-type-ish
header; ontology flavors deprioritized), and rewrites each part in place
(atomic tmp+rename) adding:

  obs.cell_type             matched label ('' for unmatched cells)
  obs.cell_type_provenance  'scea_cell_metadata:<column>' ('' unmatched)

Cell-id join: direct match of the part obs index (matrix cell id) against
the metadata id column; when <10% match directly, suffix/prefix-stripped
variants (trailing '-<digits>' stem, then the prefix before a trailing 10x
barcode -- mirroring adapter_scea.match_cells) are tried on BOTH sides of
the join. A stripped key only assigns when it resolves to exactly ONE
metadata cell_type value; ambiguous keys (e.g. an assay prefix shared by
cells of different types) never assign -- those cells stay unlabeled ('')
and are counted in the log and report. Per-experiment match rates are
logged and MERGED into scea_rescue_report.json beside --parts-dir
(subset --experiments runs update their entries without clobbering the
rest).

Parts that already have a cell_type column are skipped unless --force.

Modes (same split as the other adapters): --fetch-only downloads the
metadata TSVs into --cache on a node with EBI egress (login node) and
stops; --offline never touches the network (compute node).

Usage:
  python scea_celltype_rescue.py \
      --parts-dir /oscar/scratch/fperalta/bibm_corpus/scea/h5ad_parts --fetch-only
  python scea_celltype_rescue.py \
      --parts-dir /oscar/scratch/fperalta/bibm_corpus/scea/h5ad_parts --offline
"""
import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

# same-dir import: reuse the polite EBI fetch helper (1s throttle +
# exponential backoff on refused connections, 404/410 = permanent) plus the
# FTP layout constant and the 10x-barcode regex used by match_cells.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from adapter_scea import BARCODE_RE, FTP_BASE, fetch, open_maybe_gz  # noqa: E402

PART_RE = re.compile(r"^(?P<exp>E-[A-Z]+-\d+)\.part_\d+\.h5ad$")
EMPTY_VALUES = {"", "na", "n/a", "nan", "none", "null", "unknown",
                "not available", "not applicable"}

# column preference: authors cell type > inferred cell type > any
# cell-type-ish header; the ontology(-id) flavor of a name loses 0.5 so
# 'inferred cell type - authors labels' beats '... - ontology labels'.
CT_PATTERNS = [
    (3.0, re.compile(r"authors?[ ._-]*cell[ ._-]*type"
                     r"|cell[ ._-]*type\b.*authors?", re.I)),
    (2.0, re.compile(r"inferred[ ._-]*cell[ ._-]*type", re.I)),
    (1.0, re.compile(r"cell[ ._-]*type", re.I)),
]
ID_HEADER_RE = re.compile(r"^(cell[ ._-]?)?id$", re.I)
SUFFIX_RE = re.compile(r"-\d+$")


def log(msg):
    print(f"[scea_rescue] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# cell_metadata.tsv


def fetch_metadata(exp, cache, offline):
    for suf in (".cell_metadata.tsv", ".cell_metadata.tsv.gz"):
        p = fetch(f"{FTP_BASE}/{exp}/{exp}{suf}", cache / f"{exp}{suf}", offline)
        if p is not None:
            return p
    return None


def pick_celltype_column(header):
    """Best cell-type column index in the header, or None."""
    best_score, best_i = 0.0, None
    for i, h in enumerate(header):
        for score, pat in CT_PATTERNS:
            if pat.search(h):
                s = score - (0.5 if "ontology" in h.lower() else 0.0)
                if s > best_score:
                    best_score, best_i = s, i
                break
    return best_i


def load_celltype_map(path):
    """Parse <EXP>.cell_metadata.tsv -> ({cell_id: cell_type}, column_name)
    or (None, None) when no usable column exists."""
    with open_maybe_gz(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return None, None
        ct_i = pick_celltype_column(header)
        if ct_i is None:
            return None, None
        id_i = next((i for i, h in enumerate(header)
                     if ID_HEADER_RE.match(h.strip())), 0)
        mapping = {}
        for row in reader:
            if len(row) <= ct_i or len(row) <= id_i:
                continue
            cid, ct = row[id_i].strip(), row[ct_i].strip()
            if cid and ct.lower() not in EMPTY_VALUES:
                mapping.setdefault(cid, ct)
    return mapping, header[ct_i].strip()


# ---------------------------------------------------------------------------
# id matching (adapter_scea.match_cells stripping order)


def stripped_variants(cid):
    """Fallback id variants: trailing '-<digits>' stem, then the prefix
    before a trailing 10x barcode."""
    out = []
    stem = SUFFIX_RE.sub("", cid)
    if stem != cid:
        out.append(stem)
    if "-" in stem:
        prefix, suffix = stem.rsplit("-", 1)
        if BARCODE_RE.match(suffix):
            out.append(prefix)
    return out


def relaxed_lookup(mapping):
    """mapping re-keyed by the stripped variants of ITS OWN ids, so stripping
    can rescue the join from the metadata side too. cell_type is PER-CELL, so
    a stripped key is kept ONLY when every metadata cell producing it carries
    the same cell_type; conflicting keys go to `ambiguous` and never assign
    (an assay prefix must not fabricate one cell's label onto another).
    Returns (alt, ambiguous)."""
    alt, ambiguous = {}, set()
    for k, v in mapping.items():
        for kk in stripped_variants(k):
            if kk in ambiguous:
                continue
            if kk not in alt:
                alt[kk] = v
            elif alt[kk] != v:
                del alt[kk]
                ambiguous.add(kk)
    return alt, ambiguous


def match_ids(ids, mapping):
    """ids -> ({cell_id: cell_type}, method, n_ambiguous). Direct join first;
    when it covers <10% (adapter_scea rescue threshold), suffix/prefix-
    stripped variants are tried on both sides -- but a stripped key only
    assigns when it resolves to exactly ONE metadata cell_type value. Cells
    whose only candidate keys are ambiguous stay unlabeled and are counted
    in n_ambiguous."""
    hits = {cid: mapping[cid] for cid in ids if cid in mapping}
    method = "direct"
    n_ambiguous = 0
    if len(hits) < 0.10 * max(len(ids), 1):
        alt, ambiguous = relaxed_lookup(mapping)
        best = dict(hits)
        for cid in ids:
            if cid in best:
                continue
            hit_ambiguous = False
            for v in [cid] + stripped_variants(cid):
                ct = mapping.get(v)
                if ct is None:
                    if v in ambiguous:
                        hit_ambiguous = True
                        continue
                    ct = alt.get(v)
                if ct is not None:
                    best[cid] = ct
                    break
            else:
                if hit_ambiguous:
                    n_ambiguous += 1
        if len(best) > len(hits):
            hits, method = best, "stripped"
    return hits, method, n_ambiguous


# ---------------------------------------------------------------------------
# part rewrite


def part_has_celltype(fp):
    import h5py

    with h5py.File(fp, "r") as f:
        obs = f.get("obs")
        return obs is not None and "cell_type" in obs


def annotate_part(fp, mapping, colname, force):
    """Rewrite one part (atomic tmp+rename) with cell_type +
    cell_type_provenance. Returns (n_matched, n_cells, method, n_ambiguous),
    or None when the part already has cell_type and --force is not set.
    Matched/provenance tests use != '' so a literal '0' label survives."""
    import anndata

    if not force and part_has_celltype(fp):
        log(f"{fp.name}: cell_type already present, skipping (--force overrides)")
        return None
    a = anndata.read_h5ad(fp)
    ids = [str(x) for x in a.obs_names]
    hits, method, n_ambiguous = match_ids(ids, mapping)
    prov = f"scea_cell_metadata:{colname}"
    cts = [hits.get(cid, "") for cid in ids]
    a.obs["cell_type"] = cts
    a.obs["cell_type_provenance"] = [prov if c != "" else "" for c in cts]
    tmp = fp.with_suffix(".h5ad.tmp")
    a.write_h5ad(tmp)
    tmp.rename(fp)
    return sum(c != "" for c in cts), len(ids), method, n_ambiguous


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts-dir", type=pathlib.Path, required=True,
                    help="dir with <EXP>.part_XXXX.h5ad parts "
                         "(rewritten in place)")
    ap.add_argument("--cache", type=pathlib.Path, default=None,
                    help="metadata download cache (default: <parts-dir>/../cache)")
    ap.add_argument("--experiments", type=str, default=None,
                    help="comma-separated accession subset "
                         "(default: every experiment with parts)")
    ap.add_argument("--force", action="store_true",
                    help="re-annotate parts that already have a cell_type column")
    ap.add_argument("--fetch-only", action="store_true",
                    help="download the cell_metadata TSVs into --cache and "
                         "stop (login node); process later with --offline")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; fail on cache misses")
    args = ap.parse_args(argv)

    cache = args.cache or args.parts_dir.parent / "cache"
    parts_by_exp = defaultdict(list)
    for p in sorted(args.parts_dir.glob("*.h5ad")):
        m = PART_RE.match(p.name)
        if m:
            parts_by_exp[m.group("exp")].append(p)
    only = ({e.strip() for e in args.experiments.split(",") if e.strip()}
            if args.experiments else None)
    exps = sorted(e for e in parts_by_exp if only is None or e in only)
    if not exps:
        raise SystemExit(f"[scea_rescue] no matching parts under {args.parts_dir}")
    log(f"{len(exps)} experiments with parts: {', '.join(exps)}")

    report = {"tool": "scea_celltype_rescue", "per_experiment": {}}
    n_bad = 0
    for exp in exps:
        rep = report["per_experiment"][exp] = {"n_parts": len(parts_by_exp[exp])}
        meta = fetch_metadata(exp, cache, args.offline)
        if meta is None:
            rep["status"] = "ERROR_no_cell_metadata"
            log(f"{exp}: cell_metadata TSV unavailable"
                + (" (not in cache; run --fetch-only first)" if args.offline else ""))
            n_bad += 1
            continue
        if args.fetch_only:
            rep["status"] = "fetched"
            log(f"{exp}: metadata cached -> {meta}")
            continue
        mapping, col = load_celltype_map(meta)
        if not mapping:
            rep["status"] = "no_celltype_column"
            log(f"{exp}: no usable cell-type column in {meta.name}")
            continue
        rep["column"] = col
        rep["n_metadata_cells"] = len(mapping)
        n_match = n_cells = n_skipped = n_ambiguous = 0
        methods = set()
        failed = False
        for fp in parts_by_exp[exp]:
            try:
                res = annotate_part(fp, mapping, col, args.force)
            except Exception as e:  # noqa: BLE001 - keep going, record failure
                log(f"{fp.name}: FAILED: {e!r}")
                failed = True
                continue
            if res is None:
                n_skipped += 1
                continue
            m_, c_, method, amb_ = res
            n_match += m_
            n_cells += c_
            n_ambiguous += amb_
            methods.add(method)
            log(f"{fp.name}: {m_:,}/{c_:,} cells matched ({method})"
                + (f", {amb_:,} left unlabeled on ambiguous stripped keys"
                   if amb_ else ""))
        rep.update(n_parts_skipped=n_skipped, n_cells=n_cells,
                   n_matched=n_match, match_methods=sorted(methods),
                   n_cells_ambiguous_stripped=n_ambiguous,
                   match_rate=round(n_match / n_cells, 4) if n_cells else None)
        if failed:
            rep["status"] = "ERROR_part_failure"
            n_bad += 1
        else:
            rep["status"] = "ok"
        if n_cells:
            log(f"{exp}: MATCH RATE {n_match:,}/{n_cells:,} "
                f"({n_match / n_cells:.1%}) via {sorted(methods)} "
                f"column='{col}'"
                + (f"; {n_ambiguous:,} cells unlabeled on ambiguous "
                   f"stripped keys" if n_ambiguous else ""))
        elif n_skipped:
            log(f"{exp}: all {n_skipped} parts already annotated (skipped)")

    if not args.fetch_only:
        # merge into any existing report so subset --experiments runs update
        # their entries without clobbering earlier experiments' records
        rp = args.parts_dir.parent / "scea_rescue_report.json"
        merged = {}
        if rp.exists():
            try:
                with open(rp) as f:
                    merged = json.load(f).get("per_experiment", {}) or {}
            except Exception as e:  # noqa: BLE001 - unreadable -> start fresh
                log(f"existing report unreadable ({e!r}); starting fresh")
                merged = {}
        merged.update(report["per_experiment"])
        report["per_experiment"] = {k: merged[k] for k in sorted(merged)}
        tmp = rp.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(report, f, indent=1)
        tmp.rename(rp)
        log(f"report -> {rp} ({len(report['per_experiment'])} experiments)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
