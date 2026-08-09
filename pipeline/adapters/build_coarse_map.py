"""Build the cell-type coarsening map (DRAFT for eyeball pass).

Input : celltype_by_group_raw.csv  (cell_type, group, cells)
Output: celltype_coarse_map.csv    (cell_type, coarse_label, cells)  -- every observed value
        celltype_coarse_summary.csv (coarse_label, family, cells, share_pct, + per-group cells)
"""
import pandas as pd

BASE = "/private/tmp/claude-501/-Users-fperaltacastro-scfm-release/5bd13e3c-42df-40d8-8644-bb7cdbe852fd/scratchpad/adapters"
raw = pd.read_csv(f"{BASE}/celltype_by_group_raw.csv")
totals = raw.groupby("cell_type").cells.sum()

# ---------------------------------------------------------------- exact overrides
EXACT = {
    # ---- junk / unresolvable / tiny coherent families folded into 'other'
    "unknown": "other", "cell": "other", "abnormal cell": "other",
    "malignant cell": "other", "neoplastic cell": "other",
    "stem cell": "other", "progenitor cell": "other", "embryonic stem cell": "other",
    "mesenchymal cell": "other",
    # generic immune labels too ambiguous to train on
    "lymphocyte": "other", "leukocyte": "other", "myeloid cell": "other",
    "myeloid leukocyte": "other", "mononuclear leukocyte": "other",
    "mononuclear phagocyte": "other", "peripheral blood mononuclear cell": "other",
    "professional antigen presenting cell": "other", "inflammatory cell": "other",
    "hematopoietic cell": "other",
    # HSPC (0.09% -- too thin for its own class; candidate to resurrect)
    "cord blood hematopoietic stem cell": "other", "hematopoietic stem cell": "other",
    "hematopoietic precursor cell": "other", "common myeloid progenitor": "other",
    "common lymphoid progenitor": "other", "myeloid lineage restricted progenitor cell": "other",
    # germ & gonadal (0.08% total -- folded; candidate to resurrect)
    "spermatid": "other", "spermatocyte": "other", "spermatogonium": "other",
    "male germ cell": "other", "primordial germ cell": "other", "oocyte": "other",
    "Sertoli cell": "other", "Leydig cell": "other",
    # ---- immune specifics
    "hepatic pit cell": "NK_ILC",                       # liver-resident NK
    "Langerhans cell": "dendritic_cell",
    "follicular dendritic cell": "stromal_other",       # FDC is stromal, not hematopoietic
    "Kupffer cell": "macrophage",
    "Be cell": "B_cell",                                # effector B
    "platelet": "erythroid_megakaryocyte",
    "basophil mast progenitor cell": "granulocyte_mast",
    "CD8aa(I) thymocyte": "T_other",
    "double negative T regulatory cell": "T_other",
    "pro-T cell": "T_other",
    "cytotoxic T cell": "T_other",                      # generic, no CD4/CD8 info
    # ---- neural
    "Cajal-Retzius cell": "neural_other",
    "Bergmann glial cell": "astrocyte",
    "cerebellar granule cell": "neuron_excitatory",
    "OFFx cell": "retinal_ganglion",                    # RGC type in human retina taxonomies (CHECK)
    "Mueller cell": "mueller_glia",
    # ---- ocular
    "pigmented ciliary epithelial cell": "ocular_anterior_segment",
    "non-pigmented ciliary epithelial cell": "ocular_anterior_segment",
    "conjunctival epithelial cell": "ocular_anterior_segment",
    "corneal epithelial cell": "ocular_anterior_segment",
    "corneal endothelial cell": "ocular_anterior_segment",   # NOT vascular endothelium
    "keratocyte": "ocular_anterior_segment",                 # corneal stroma
    "trabecular meshwork cell": "ocular_anterior_segment",
    "pigmented epithelial cell": "ocular_anterior_segment",  # iris/ciliary PE in eye datasets (CHECK)
    "eye photoreceptor cell": "photoreceptor",
    "ciliary muscle cell": "smooth_muscle",                  # it IS smooth muscle (eye tissue)
    # ---- stromal / vascular / muscle
    "interstitial cell of Cajal": "stromal_other",
    "mesangial cell": "pericyte_perivascular",
    "adventitial cell": "pericyte_perivascular",
    "brain vascular cell": "pericyte_perivascular",
    "vascular leptomeningeal cell": "pericyte_perivascular",
    "peritubular myoid cell": "smooth_muscle",
    "myometrial cell": "smooth_muscle",
    "vascular lymphangioblast": "endothelial",
    "endothelial tip cell": "endothelial",
    "decidual cell": "stromal_other", "granulosa cell": "stromal_other",
    "theca cell": "stromal_other", "chondrocyte": "stromal_other",
    "tendon cell": "stromal_other", "reticular cell": "stromal_other",
    "connective tissue cell": "stromal_other", "kidney interstitial cell": "stromal_other",
    "fibrocyte": "fibroblast",
    "preadipocyte": "adipocyte_MSC",
    "fibro/adipogenic progenitor cell": "adipocyte_MSC",
    "muscle cell": "striated_muscle",                   # generic (CHECK -- could be smooth)
    "tongue muscle cell": "striated_muscle",
    # ---- epithelial, ambiguous generics
    "goblet cell": "epithelial_other",                  # airway? gut? conjunctiva? (CHECK)
    "secretory cell": "epithelial_other",
    "serous secreting cell": "epithelial_other",        # salivary vs airway (CHECK)
    "acinar cell": "epithelial_other",
    "duct epithelial cell": "epithelial_other",
    "mucus secreting cell": "epithelial_other",
    "myoepithelial cell": "epithelial_other",
    "seromucus secreting cell": "epithelial_airway",    # submucosal gland (CHECK)
    "peg cell": "epithelial_other",                     # fallopian tube
    "thyroid follicular cell": "epithelial_other",
    "endocrine cell": "epithelial_other", "neuroendocrine cell": "epithelial_other",
    "Merkel cell": "epithelial_other", "taste receptor cell": "epithelial_other",
    "supporting cell of vestibular epithelium": "epithelial_other",
    "vestibular dark cell": "epithelial_other",
    "luminal endometrial multiciliated epithelial cell": "epithelial_other",
    "neuro-medullary thymic epithelial cell": "epithelial_other",
    "medullary thymic epithelial cell": "epithelial_other",
    "salivary gland cell": "epithelial_other",
    "parietal epithelial cell": "epithelial_kidney_urogenital",  # kidney (Bowman's capsule)
    "podocyte": "epithelial_kidney_urogenital",
    "PP cell": "epithelial_pancreas",
    "lactocyte": "epithelial_mammary",
    "sebocyte": "epithelial_squamous_basal",
    "enterochromaffin-like cell": "epithelial_GI",
    "peptic cell": "epithelial_GI",
}

# ---------------------------------------------------------------- ordered rules
def classify(name: str) -> str:
    if name in EXACT:
        return EXACT[name]
    l = name.lower()
    has = lambda *ks: any(k in l for k in ks)
    # retina first (before generic neuron/GABAergic/endothelial rules)
    if "amacrine" in l: return "amacrine"
    if "bipolar" in l: return "retinal_bipolar"
    if "ganglion cell" in l: return "retinal_ganglion"
    if "horizontal" in l: return "horizontal_cell"
    if has("photoreceptor", "retinal rod", "retinal cone") or l == "s cone cell":
        return "photoreceptor"
    if "retinal pigment epithelial" in l: return "RPE"
    # myeloid / lymphoid
    if has("microglial", "central nervous system macrophage", "meningeal macrophage"):
        return "microglia"
    if "dendritic" in l: return "dendritic_cell"
    if "monocyte" in l: return "monocyte"
    if "macrophage" in l: return "macrophage"
    if has("mast cell", "neutrophil", "basophil", "eosinophil", "granulocyte"):
        return "granulocyte_mast"
    if has("erythro", "megakaryocyte"): return "erythroid_megakaryocyte"
    if has("plasma cell", "plasmablast"): return "plasma_cell"
    if " b cell" in f" {l}": return "B_cell"
    if "natural killer" in l or "innate lymphoid" in l: return "NK_ILC"
    if "nk t cell" in l: return "T_other"                       # NKT are T cells
    if (" t cell" in f" {l}" or has("thymocyte", "t-helper", "t follicular helper",
                                    "t regulatory")):
        l2 = l.replace("cd45ro", "").replace("cd45ra", "").replace("cd45", "")
        if "gamma-delta" in l: return "T_other"
        if "cd4" in l2 and "cd4-negative" not in l2: return "CD4_T"
        if "cd8" in l2 and "cd8-negative" not in l2: return "CD8_T"
        if has("regulatory", "helper"): return "CD4_T"          # unspecified Treg/helper -> CD4
        return "T_other"
    # neural / glia (before endothelial so nothing neural leaks; enteroglial before GI)
    if "schwann" in l: return "schwann_cell"
    if "oligodendrocyte" in l: return "oligodendrocyte_lineage"
    if "astrocyte" in l: return "astrocyte"
    if "glutamatergic" in l: return "neuron_excitatory"
    if has("gabaergic", "interneuron", "medium spiny"): return "neuron_inhibitory"
    if "neuron" in l: return "neural_other"
    if has("glial", "glia", "ependymal", "neural", "retinal progenitor"):
        return "neural_other"
    # vasculature & stroma
    if "endothelial" in l: return "endothelial"
    if has("fibroblast", "myofibroblast", "stellate"): return "fibroblast"
    if has("pericyte", "perivascular", "mural"): return "pericyte_perivascular"
    if "mesothelial" in l: return "mesothelial"
    if "smooth muscle" in l: return "smooth_muscle"
    if has("cardiac muscle", "cardiac myocyte", "skeletal muscle",
           "muscle fiber", "fast muscle", "slow muscle", "satellite"):
        return "striated_muscle"
    if "adipocyte" in l: return "adipocyte_MSC"
    if "mesenchymal stem cell" in l: return "adipocyte_MSC"
    if "melanocyte" in l: return "melanocyte"
    if "stromal" in l: return "stromal_other"
    # epithelial families
    if "alveolar" in l: return "epithelial_alveolar"
    if has("mammary", "breast"): return "epithelial_mammary"
    if has("kidney", "renal", "loop of henle", "tubule", "urothel", "urethra",
           "bladder", "prostate", "collecting duct"):
        return "epithelial_kidney_urogenital"
    if has("entero", "colon", "intestin", "duoden", "ileum", "paneth", "tuft",
           "crypt", "gut", "microfold", "transit amplifying", "best4", "gastric"):
        return "epithelial_GI"
    if has("hepatocyte", "cholangiocyte"): return "hepatocyte_biliary"
    if "pancrea" in l: return "epithelial_pancreas"
    if has("respiratory", "trachea", "bronch", "club cell", "ciliated", "nasal",
           "airway", "lung", "pulmonary", "ionocyte", "goblet", "serous",
           "deuterosomal", "hillock", "brush cell"):
        return "epithelial_airway"
    if has("keratinocyte", "epidermis", "stratified", "squamous", "keratinizing",
           "stratum germinativum") or l == "basal cell":
        return "epithelial_squamous_basal"
    if has("epithelial", "epithelium", "salivary"): return "epithelial_other"
    return "UNMAPPED"

FAMILY = {
    "CD4_T": "immune", "CD8_T": "immune", "T_other": "immune", "NK_ILC": "immune",
    "B_cell": "immune", "plasma_cell": "immune", "monocyte": "immune",
    "macrophage": "immune", "dendritic_cell": "immune", "microglia": "immune",
    "granulocyte_mast": "immune", "erythroid_megakaryocyte": "immune",
    "fibroblast": "stromal", "stromal_other": "stromal", "adipocyte_MSC": "stromal",
    "pericyte_perivascular": "stromal", "endothelial": "stromal",
    "smooth_muscle": "stromal", "striated_muscle": "stromal", "mesothelial": "stromal",
    "schwann_cell": "neural", "oligodendrocyte_lineage": "neural", "astrocyte": "neural",
    "neuron_excitatory": "neural", "neuron_inhibitory": "neural", "neural_other": "neural",
    "photoreceptor": "ocular", "retinal_bipolar": "ocular", "amacrine": "ocular",
    "retinal_ganglion": "ocular", "horizontal_cell": "ocular", "mueller_glia": "ocular",
    "RPE": "ocular", "ocular_anterior_segment": "ocular", "melanocyte": "ocular",
    "epithelial_alveolar": "epithelial", "epithelial_airway": "epithelial",
    "epithelial_GI": "epithelial", "hepatocyte_biliary": "epithelial",
    "epithelial_pancreas": "epithelial", "epithelial_mammary": "epithelial",
    "epithelial_kidney_urogenital": "epithelial", "epithelial_squamous_basal": "epithelial",
    "epithelial_other": "epithelial",
    "other": "other",
}

mapping = {ct: classify(ct) for ct in totals.index}
un = {ct: totals[ct] for ct, c in mapping.items() if c == "UNMAPPED"}
if un:
    print("UNMAPPED:")
    for k, v in sorted(un.items(), key=lambda x: -x[1]):
        print(f"  {k}\t{v:,}")
    raise SystemExit(1)

# ---- write per-cell_type map (every observed value)
m = pd.DataFrame({"cell_type": totals.index, "coarse_label": [mapping[c] for c in totals.index],
                  "cells": totals.values})
m = m.sort_values(["coarse_label", "cells"], ascending=[True, False])
m.to_csv(f"{BASE}/celltype_coarse_map.csv", index=False)

# ---- summary
raw["coarse_label"] = raw.cell_type.map(mapping)
tot = raw.cells.sum()
cls = raw.groupby("coarse_label").cells.sum().sort_values(ascending=False)
bygrp = raw.pivot_table(index="coarse_label", columns="group", values="cells",
                        aggfunc="sum", fill_value=0)
summ = pd.DataFrame({"coarse_label": cls.index,
                     "family": [FAMILY[c] for c in cls.index],
                     "cells": cls.values,
                     "share_pct": (100 * cls / tot).round(3).values})
summ = summ.merge(bygrp, left_on="coarse_label", right_index=True)
summ.to_csv(f"{BASE}/celltype_coarse_summary.csv", index=False)

named = cls.drop("other")
print(f"classes: {len(cls)} total = {len(named)} named + 'other'")
print(f"cells: {tot:,}   named-class coverage: {100*named.sum()/tot:.2f}%   'other': {100*cls['other']/tot:.2f}%")
print(f"named class share: min {named.min():,} ({named.idxmin()}, {100*named.min()/tot:.3f}%)"
      f"  max {named.max():,} ({named.idxmax()}, {100*named.max()/tot:.2f}%)")

print("\n=== per class (overall) ===")
for c, n in cls.items():
    print(f"{c:<30}{FAMILY[c]:<12}{n:>12,}{100*n/tot:>8.2f}%")

print("\n=== per macro-group: total cells, named classes absent (0 cells) / thin (<1000) ===")
gt = raw.groupby("group").cells.sum().sort_values(ascending=False)
named_cols = [c for c in cls.index if c != "other"]
for g in gt.index:
    col = bygrp[g]
    absent = [c for c in named_cols if col.get(c, 0) == 0]
    thin = [c for c in named_cols if 0 < col.get(c, 0) < 1000]
    gshare = col / col.sum() * 100
    gn = gshare.drop("other", errors="ignore")
    gn = gn[gn > 0]
    print(f"\n{g}: {gt[g]:,} cells | class share min {gn.min():.3f}% ({gn.idxmin()}) "
          f"max {gn.max():.1f}% ({gn.idxmax()})")
    print(f"  ABSENT ({len(absent)}): {', '.join(absent) if absent else '-'}")
    print(f"  thin<1k ({len(thin)}): {', '.join(thin) if thin else '-'}")
