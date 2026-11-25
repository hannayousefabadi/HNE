"""scripts/additional_signatures/add_fmrp_module_scores.py

Runs on the server (needs the h5ads). Adds Seurat-style module scores (module_scores.module_scores) of the FMRP
signature variants (gene_sets.load_fmrp_variants) and of APM4 / APM12 to the spot tables that
extract_spot_signatures.py wrote: one column "{set}_module_score" per set, e.g. FMRP_bulk_up_module_score,
APM4_module_score. The ssGSEA columns and everything else in the tables stay as they are.
Scored on the in-tissue spots of each patient. Push the tables afterwards; plot_fmrp_module_scores.py runs
locally from them. A table that already has all the columns is skipped.
"""
import json
import logging

import pandas as pd

from hne.additional_signatures.gene_sets import APM4_GENES, APM12_GENES, load_fmrp_variants
from hne.additional_signatures.module_scores import module_scores
from hne.additional_signatures.paths import SPOT_TABLES
from hne.core.data_io import load_visium
from hne.core.paths import PATIENTS

CONFIG = {"n_bins": 25, "ctrl_size": 50, "seed": 0}     # scanpy.tl.score_genes defaults, seed fixed

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

gene_sets = {**load_fmrp_variants(), "APM4": APM4_GENES, "APM12": APM12_GENES}
columns = [f"{name}_module_score" for name in gene_sets]
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
    scores, genes_used = module_scores(vis, gene_sets, CONFIG["n_bins"], CONFIG["ctrl_size"], CONFIG["seed"])
    for name in gene_sets:
        table[f"{name}_module_score"] = scores.loc[table["barcode"], name].to_numpy(dtype=float)
    table.to_csv(path, index=False)
    (SPOT_TABLES / f"{patient_id}_module_genes_used.json").write_text(json.dumps(genes_used))
    print(f"[{i}/{len(tables)}] {patient_id}: genes used {genes_used}")
