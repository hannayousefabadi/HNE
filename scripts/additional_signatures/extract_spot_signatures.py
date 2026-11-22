"""scripts/additional_signatures/extract_spot_signatures.py

Step 2. For each selected naive patient, writes results/additional_signatures/spot_tables/{patient}.csv:
one row per in-tissue spot with tumor fraction, FMRP / APM4 / APM12 scores and the immune count
within the radius. Finished patients are skipped, so a rerun resumes.
"""
import json
import logging
import traceback

import pandas as pd

from hne.additional_signatures.paths import NAIVE_PATIENTS_CSV, SPOT_TABLES
from hne.additional_signatures.spot_extraction import extract_patient_spots

CONFIG = {
    "radius_um": 200.0,          # neighbourhood in which immune cells are counted
    "include_self": True,        # the spot's own immune cells count as inside its radius
    "ssgsea_processes": 4,
    "ssgsea_min_size": 1,
}

for noisy in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

selected = pd.read_csv(NAIVE_PATIENTS_CSV).query("selected")["patient_id"].tolist()
if not selected:
    raise RuntimeError(f"no selected patient in {NAIVE_PATIENTS_CSV}, run select_naive_patients.py")
(SPOT_TABLES / "run_config.json").write_text(json.dumps(CONFIG, indent=2))

failed = {}
for i, patient_id in enumerate(selected, 1):
    out = SPOT_TABLES / f"{patient_id}.csv"
    if out.exists():
        print(f"[{i}/{len(selected)}] {patient_id}: done already")
        continue
    try:
        table, genes_used = extract_patient_spots(patient_id, CONFIG["radius_um"], CONFIG["include_self"],
                                                  CONFIG["ssgsea_processes"], CONFIG["ssgsea_min_size"])
    except Exception:
        failed[patient_id] = traceback.format_exc()
        print(f"[{i}/{len(selected)}] {patient_id}: FAILED")
        continue
    table.to_csv(out, index=False)
    (SPOT_TABLES / f"{patient_id}_genes_used.json").write_text(json.dumps(genes_used))
    print(f"[{i}/{len(selected)}] {patient_id}: {len(table)} spots, genes used {genes_used}")

failures_file = SPOT_TABLES / "failures.txt"
if failed:
    failures_file.write_text("\n".join(f"=== {patient_id}\n{trace}" for patient_id, trace in failed.items()))
    for patient_id, trace in failed.items():
        print(f"\n=== {patient_id}\n{trace}")
    print(f"\ntracebacks also written to {failures_file}")
    raise RuntimeError(f"{len(failed)} patient(s) failed: {sorted(failed)}")
