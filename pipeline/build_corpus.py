"""BIBM balanced-corpus builder.

Stage A (metadata): pull obs for all labeled, healthy, primary human cells;
macro-group; donor-cap 10k; sample the four arm memberships + donor holdout.
Stage B (expression): fetch raw counts for selected cells in chunks -> h5ad parts.

Resumable: Stage A writes selection.parquet once; Stage B skips existing parts.
"""
import argparse
import collections
import pathlib

import cellxgene_census
import numpy as np
import pandas as pd

CENSUS = "2025-11-08"
SEED = 42
CAP = 10_000
HOLDOUT_DONOR_FRAC = 0.10
CHUNK = 100_000

OUT = pathlib.Path("/oscar/scratch/fperalta/bibm_corpus")

MACRO = {}
for l in ["European American", "Australian", "British", "German", "Irish", "Finnish"]:
    MACRO[l] = "European"
for l in ["African American", "Ethiopian"]:
    MACRO[l] = "African"
for l in ["Hispanic or Latin", "Brazilian"]:
    MACRO[l] = "HispanicLatino"
for l in ["Asian", "Han Chinese", "Korean", "Japanese", "Singaporean Chinese",
          "Singaporean Malay", "Thai", "Chinese", "East Asian", "Southeast Asian"]:
    MACRO[l] = "EastSEAsian"
for l in ["South Asian", "Indian", "Singaporean Indian", "Bangladeshi"]:
    MACRO[l] = "SouthAsian"
GROUPS = ["African", "HispanicLatino", "SouthAsian", "EastSEAsian", "European"]

VF = "self_reported_ethnicity != 'unknown' and is_primary_data == True and disease == 'normal'"
OBS_COLS = ["soma_joinid", "dataset_id", "donor_id", "tissue_general",
            "self_reported_ethnicity", "cell_type", "assay"]


def stage_a():
    rng = np.random.default_rng(SEED)
    with cellxgene_census.open_soma(census_version=CENSUS) as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        obs = census["census_data"]["homo_sapiens"].obs
        parts = [b.to_pandas() for b in obs.read(column_names=OBS_COLS, value_filter=VF)]
    df = pd.concat(parts, ignore_index=True)
    ds_meta = ds[["dataset_id", "collection_id", "collection_name",
                  "collection_doi", "dataset_title"]].drop_duplicates("dataset_id")
    df = df.merge(ds_meta[["dataset_id", "collection_id"]], on="dataset_id")
    df["group"] = df.self_reported_ethnicity.map(MACRO)
    df = df.dropna(subset=["group"])
    df["donor_key"] = df.collection_id.astype(str) + "|" + df.donor_id.astype(str)
    print(f"pool: {len(df):,} cells, {df.donor_key.nunique():,} donors")

    # donor cap: sample <= CAP cells per donor
    keep_idx = []
    for _, idx in df.groupby("donor_key").indices.items():
        keep_idx.append(rng.choice(idx, size=CAP, replace=False) if len(idx) > CAP else idx)
    df = df.iloc[np.concatenate(keep_idx)].reset_index(drop=True)
    print(f"after cap {CAP}: {len(df):,} cells")

    # donor holdout per group
    df["split"] = "train"
    for g in GROUPS:
        dk = df.loc[df.group == g, "donor_key"].unique()
        hold = rng.choice(dk, size=max(2, int(len(dk) * HOLDOUT_DONOR_FRAC)), replace=False)
        df.loc[df.donor_key.isin(hold), "split"] = "eval_donor"
    train = df[df.split == "train"]

    # arm sizes from the training pool
    gsz = train.groupby("group").size()
    floor = int(gsz.min())
    tmin = train.pivot_table(index="tissue_general", columns="group",
                             values="soma_joinid", aggfunc="count", fill_value=0)[GROUPS].min(axis=1)
    tmin = tmin[tmin > 0]
    mfloor = int(tmin.sum())
    print(f"train floors: balanced={floor:,}/group  matched={mfloor:,}/group "
          f"({len(tmin)} tissues)")

    def sample_group(pool, n):
        return rng.choice(pool.index.to_numpy(), size=n, replace=False)

    df["arm_balanced"] = False
    df["arm_matched"] = False
    df["arm_proportional"] = False
    for g in GROUPS:
        pool = train[train.group == g]
        df.loc[sample_group(pool, floor), "arm_balanced"] = True
        for t, m in tmin.items():
            tp = pool[pool.tissue_general == t]
            df.loc[sample_group(tp, min(int(m), len(tp))), "arm_matched"] = True
    # proportional: same TOTAL as balanced, natural group shares of the capped pool
    total = floor * len(GROUPS)
    shares = (gsz / gsz.sum() * total).round().astype(int)
    for g in GROUPS:
        pool = train[train.group == g]
        df.loc[sample_group(pool, min(shares[g], len(pool))), "arm_proportional"] = True

    sel = df[df.arm_balanced | df.arm_matched | df.arm_proportional | (df.split == "eval_donor")]
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(OUT / "selection.parquet")
    ds_meta[ds_meta.dataset_id.isin(sel.dataset_id.unique())].to_csv(
        OUT / "MANIFEST_datasets.csv", index=False)
    summary = sel.groupby("group").agg(
        cells=("soma_joinid", "count"), donors=("donor_key", "nunique"))
    print(summary.to_string())
    print(f"selection: {len(sel):,} unique cells -> {OUT/'selection.parquet'}")


def stage_b():
    sel = pd.read_parquet(OUT / "selection.parquet")
    ids = np.sort(sel.soma_joinid.to_numpy())
    xdir = OUT / "h5ad_parts"
    xdir.mkdir(exist_ok=True)
    n_parts = int(np.ceil(len(ids) / CHUNK))
    print(f"fetching {len(ids):,} cells in {n_parts} parts of {CHUNK:,}")
    with cellxgene_census.open_soma(census_version=CENSUS) as census:
        for i in range(n_parts):
            fp = xdir / f"part_{i:04d}.h5ad"
            if fp.exists():
                continue
            chunk = ids[i * CHUNK:(i + 1) * CHUNK].tolist()
            ad = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", obs_coords=chunk,
                obs_column_names=["soma_joinid", "dataset_id", "donor_id", "tissue_general",
                                  "self_reported_ethnicity", "cell_type", "assay", "disease"])
            ad.write_h5ad(fp)
            print(f"part {i+1}/{n_parts} done ({fp.stat().st_size/1e9:.2f} GB)", flush=True)
    print("STAGE B COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["a", "b", "all"])
    a = ap.parse_args()
    if a.stage in ("a", "all"):
        stage_a()
    if a.stage in ("b", "all"):
        stage_b()
