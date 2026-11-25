"""scripts/additional_signatures/add_spot_depth.py

Runs on the server (needs the h5ads). Adds the sequencing depth of every spot to the spot tables that
extract_spot_signatures.py wrote, from the h5ad layer raw_count: columns total_counts (UMIs in the spot) and
n_genes_detected (genes with at least one count). Nothing else in the tables changes. Push the tables afterwards;
plot_fmrp_updown_depth_adjusted.py runs locally from them. A table that already has both columns is skipped.
"""
import logging

import numpy as np
import pandas as pd

from hne.additional_signatures.paths import SPOT_TABLES
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

COLUMNS = ["total_counts", "n_genes_detected"]

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

tables = sorted(SPOT_TABLES.glob("CH_L_*.csv"))
if not tables:
    raise RuntimeError(f"no spot table in {SPOT_TABLES}, run extract_spot_signatures.py")

for i, path in enumerate(tables, 1):
    patient_id = path.stem
    table = pd.read_csv(path)
    if all(c in table.columns for c in COLUMNS):
        print(f"[{i}/{len(tables)}] {patient_id}: done already")
        continue
    vis = load_visium(PATIENTS[patient_id], keep_layers=["raw_count"])
    missing = set(table["barcode"]) - set(vis.obs_names)
    if missing:
        raise KeyError(f"{patient_id}: {len(missing)} spot barcodes of the table are not in the h5ad")
    counts = vis[table["barcode"].to_numpy()].layers["raw_count"]
    table["total_counts"] = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    table["n_genes_detected"] = np.asarray((counts > 0).sum(axis=1)).ravel().astype(int)
    table.to_csv(path, index=False)
    print(f"[{i}/{len(tables)}] {patient_id}: median {table['total_counts'].median():.0f} counts, "
          f"{table['n_genes_detected'].median():.0f} genes per spot")
