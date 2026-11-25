"""src/hne/additional_signatures/gene_sets.py

Gene sets scored per spot in this analysis. Human symbols.

Both APM versions are kept so the plots can show which one behaves better.
HLA-A, HLA-B and HLA-C are not on the Visium Human Transcriptome Probe Set v2.0 (the panel of
these slides), so they are never in the h5ad and the 12-gene set is scored on its 9 measured genes.
"""
import json
from pathlib import Path

# copy of the FMRP_signature in hne.preprocessing.spot_signatures.compute_signatures, where it is a
# local variable and cannot be imported. Keep the two identical.
FMRP_GENES = ["MARCKSL1", "S100A16", "DDAH1", "MYCL", "SHANK2", "ITIH2", "PIK3AP1",
              "LHFPL6", "FRMD5", "CLDN6", "ATP11A", "SLC25A21", "B4GALNT3", "WNT10A",
              "KCTD17", "BCAM", "CCL14", "CCL15", "CCL23", "DLG4", "SPTSSB", "SOGA1",
              "MAP9", "CCDC149", "CMBL", "PTPRN", "WTIP", "FXR1", "ARHGEF26", "PROS1",
              "PARP8", "OSR1", "TFF2", "UCHL1", "PRSS35", "KCNK5", "AEBP1", "SP8", "CFTR",
              "CYSLTR1", "FSCN1", "IL33", "ELFN1", "AFAP1L1", "LPAR4", "CASD1", "HS6ST2",
              "CD109", "MAL2", "PHF19"]

# FMRP knockout signatures, copied verbatim from the R object FMRP.signature into fmrp_signature.json:
# {"bulk.signature": {"up": [...], "down": [...]}, "all.signature": {"up": [...], "down": [...]}}.
# "up" / "down" there are the genes up / down in FMRP knockout, so they are flipped here: a gene down in
# knockout is up when FMRP is active. R "bulk down" (50 genes) is the FMRP_GENES list above.
# Sizes are checked so an edited file fails loudly.
FMRP_SIGNATURE_JSON = Path(__file__).with_name("fmrp_signature.json")
FMRP_SIGNATURE_SIZES = {("bulk", "up"): 36, ("bulk", "down"): 50, ("all", "up"): 66, ("all", "down"): 90}
FMRP_ACTIVE_DIRECTION = {"down": "up", "up": "down"}   # knockout direction -> FMRP-active direction

# R symbols renamed to the symbol of the same gene in the h5ad (FMRP_GENES already uses CCL14).
# CKMT1A, CKMT1B, RPL29 and GNAS are not on the probe panel and are dropped at scoring time.
H5AD_SYMBOLS = {"CCL15-CCL14": "CCL14", "H3-3B": "H3F3B"}


def load_fmrp_variants() -> dict:
    """FMRP-active sets {"FMRP_bulk_up", "FMRP_bulk_down", "FMRP_all_up", "FMRP_all_down"}, plus a
    "{name}_noAPM" copy, without the APM4 / APM12 genes, of every set that shares genes with them."""
    if not FMRP_SIGNATURE_JSON.exists():
        raise FileNotFoundError(f"{FMRP_SIGNATURE_JSON} is missing: export FMRP.signature from R (see review/07_additional_signatures.md)")
    raw = json.loads(FMRP_SIGNATURE_JSON.read_text())
    apm = set(APM4_GENES) | set(APM12_GENES)
    variants = {}
    for (kind, ko_direction), size in FMRP_SIGNATURE_SIZES.items():
        genes = [H5AD_SYMBOLS.get(g, g) for g in raw[f"{kind}.signature"][ko_direction]]
        if len(genes) != size:
            raise ValueError(f"{kind} {ko_direction}: {len(genes)} genes in {FMRP_SIGNATURE_JSON.name}, expected {size}")
        name = f"FMRP_{kind}_{FMRP_ACTIVE_DIRECTION[ko_direction]}"
        variants[name] = genes
        if apm & set(genes):
            variants[f"{name}_noAPM"] = [g for g in genes if g not in apm]
    return variants


# 4-gene antigen-processing set
APM4_GENES = ["TAP1", "TAP2", "PSMB8", "PSMB10"]

# Dijkstra et al., Cancer Cell 2025: 11 mouse genes (Nlrc5, Erap1, Calr, Canx, Pdia3, B2m, H2-K1,
# H2-D1, Sppl3, Mogs, Ganab); H2-K1 and H2-D1 map to HLA-A, HLA-B, HLA-C, giving 12 human genes
APM12_GENES = ["NLRC5", "ERAP1", "CALR", "CANX", "PDIA3", "B2M", "HLA-A", "HLA-B", "HLA-C",
               "SPPL3", "MOGS", "GANAB"]

GENE_SETS = {
    "FMRP": FMRP_GENES,
    "APM4": APM4_GENES,
    "APM12": APM12_GENES,
}

# single genes reported as plain log1p-CPM expression per spot (one gene is not an ssGSEA set).
# Column name in the spot tables: "{gene}_expression".
SINGLE_GENES = ["FMR1"]

# deconvolution cell types counted as immune. Keys of obsm["deconv_sample_level_custom1_weights"] in the
# v2 h5ads on S3, which have no T_nos, T_reg or NK column (older local copies do), so regulatory T cells,
# NK cells and unspecified T cells are not counted unless they sit in T_CD4, T_CD8 or T_NK_cycling.
IMMUNE_CELL_TYPES = ["B_cell", "Plasma", "T_CD4", "T_CD8", "T_NK_cycling",
                     "MoMac", "Granulocyte", "Mast", "cDC1", "cDC2", "cDC_LAMP3", "pDC"]
