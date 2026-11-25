"""scripts/additional_signatures/add_fmrp_singscores.py

Runs on the server (needs the h5ads). Adds singscores (hne.additional_signatures.singscore, Foroutan 2018) of the
FMRP signature variants (gene_sets.load_fmrp_variants) and of APM4 / APM12 to the spot tables that
extract_spot_signatures.py wrote: one column "{set}_singscore" per set, e.g. FMRP_bulk_up_singscore. The singscore
TotalScore of a signature is its up column minus its down column (computed in plot_fmrp_singscores.py).
Every gene of the h5ad is ranked within each in-tissue spot. The other columns of the tables stay as they are.
Push the tables afterwards; plot_fmrp_singscores.py runs locally from them. A table that already has all the
columns is skipped.
"""
import json
import logging

import pandas as pd

from hne.additional_signatures.gene_sets import APM4_GENES, APM12_GENES, load_fmrp_variants
from hne.additional_signatures.paths import SPOT_TABLES
from hne.additional_signatures.singscore import singscores
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

gene_sets = {**load_fmrp_variants(), "APM4": APM4_GENES, "APM12": APM12_GENES}
columns = [f"{name}_singscore" for name in gene_sets]
tables = sorted(SPOT_TABLES.glob("CH_L_*.csv"))
if not tables:
    raise RuntimeError(f"no spot table in {SPOT_TABLES}, run extract_spot_signatures.py")

for i, path in enumerate(tables, 1):
    patient_id = path.stem
    table = pd.read_csv(path)
    if all(c in table.columns for c in columns):
        print(f"[{i}/{len(tables)}] {patient_id}: done already")
        continue
    vis = load_visium(PATIENTS[patient_id], keep_layers=["log_norm_count"])
    vis = vis[vis.obs["in_tissue"].astype(int) == 1].copy()
    missing = set(table["barcode"]) - set(vis.obs_names)
    if missing:
        raise KeyError(f"{patient_id}: {len(missing)} spot barcodes of the table are not in-tissue spots of the h5ad")
    scores, genes_used = singscores(vis, gene_sets)
    for name in gene_sets:
        table[f"{name}_singscore"] = scores.loc[table["barcode"], name].to_numpy(dtype=float)
    table.to_csv(path, index=False)
    (SPOT_TABLES / f"{patient_id}_singscore_genes_used.json").write_text(json.dumps(genes_used))
    print(f"[{i}/{len(tables)}] {patient_id}: genes used {genes_used}")
