"""scripts/additional_signatures/add_fmrp_variant_scores.py

Runs on the server (needs the h5ads). Adds the ssGSEA scores of the four FMRP signature variants
(gene_sets.load_fmrp_variants: bulk / all, up / down) to the spot tables that extract_spot_signatures.py
already wrote: columns FMRP_bulk_up_score, FMRP_bulk_down_score, FMRP_all_up_score, FMRP_all_down_score.
Nothing else in the tables changes. Push the tables afterwards; plot_fmrp_variant_maps.py runs locally from them.
A table that already has all four columns is skipped.
"""
import json
import logging

import pandas as pd

from hne.additional_signatures.gene_sets import load_fmrp_variants
from hne.additional_signatures.paths import SPOT_TABLES
from hne.additional_signatures.spot_extraction import score_gene_sets
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

CONFIG = {"ssgsea_processes": 4, "ssgsea_min_size": 1}      # same as extract_spot_signatures.py

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

variants = load_fmrp_variants()
columns = [f"{name}_score" for name in variants]
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
    scores, genes_used = score_gene_sets(vis, CONFIG["ssgsea_processes"], CONFIG["ssgsea_min_size"], gene_sets=variants)
    missing = set(table["barcode"]) - set(scores.index)
    if missing:
        raise KeyError(f"{patient_id}: {len(missing)} spot barcodes of the table are not in the h5ad")
    for name in variants:
        table[f"{name}_score"] = scores.loc[table["barcode"], name].to_numpy(dtype=float)
    table.to_csv(path, index=False)
    (SPOT_TABLES / f"{patient_id}_fmrp_variants_genes_used.json").write_text(json.dumps(genes_used))
    print(f"[{i}/{len(tables)}] {patient_id}: genes used {genes_used}")
