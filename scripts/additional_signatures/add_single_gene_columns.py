"""scripts/additional_signatures/add_single_gene_columns.py

Adds the single-gene expression columns (gene_sets.SINGLE_GENES, e.g. FMR1_expression) to the spot tables
that extract_spot_signatures.py already wrote, without rescoring anything. Loads each patient's h5ad once.
A table that already has all the columns is skipped.
"""
import logging

import pandas as pd

from hne.additional_signatures.gene_sets import SINGLE_GENES
from hne.additional_signatures.paths import SPOT_TABLES
from hne.additional_signatures.spot_extraction import single_gene_expression
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

columns = [f"{g}_expression" for g in SINGLE_GENES]
tables = sorted(p for p in SPOT_TABLES.glob("CH_L_*.csv"))
if not tables:
    raise RuntimeError(f"no spot table in {SPOT_TABLES}, run extract_spot_signatures.py")

for i, path in enumerate(tables, 1):
    patient_id = path.stem
    table = pd.read_csv(path)
    if all(c in table.columns for c in columns):
        print(f"[{i}/{len(tables)}] {patient_id}: done already")
        continue
    vis = load_visium(PATIENTS[patient_id], keep_layers=["log_norm_count"])
    expression = single_gene_expression(vis, SINGLE_GENES)
    missing = set(table["barcode"]) - set(expression.index)
    if missing:
        raise KeyError(f"{patient_id}: {len(missing)} spot barcodes of the table are not in the h5ad")
    for column in columns:
        table[column] = expression.loc[table["barcode"], column].to_numpy(dtype=float)
    table.to_csv(path, index=False)
    print(f"[{i}/{len(tables)}] {patient_id}: " + ", ".join(f"{c} detected in {(table[c] > 0).mean():.1%} of spots" for c in columns))
