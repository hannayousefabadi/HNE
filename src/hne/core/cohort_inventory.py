"""src/core/cohort_inventory.py"""

import re
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from hne.core.s3_io import S3DataLoader
from hne.core.paths import (RESULTS, RAW_DATA_BUCKET, RAW_DATA_PREFIX, 
                            PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX)

loader = S3DataLoader()

# CH_L_<digits><optional letter>
PATIENT_ID_PATTERN = re.compile(r"CH_L_\d+[a-z]?")

def list_raw_hne(bucket, prefix):
    """List every .tiff of H&E images"""
    full_prefix = f"{prefix.rstrip('/')}/spatial_transcriptomics/Visium/image_files/"
    paginator = loader.s3_client.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.tif'):
                files.append(key)

    return files            


def list_processed_visium(bucket, prefix):
    """List every patient folder from processed Visium data"""
    full_prefix = f"{prefix.rstrip('/')}/v2/spaceranger_count/"
    paginator = loader.s3_client.get_paginator('list_objects_v2')
    patients = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter='/'):
        for common in page.get('CommonPrefixes', []):
            folder = common['Prefix'].rstrip('/').split('/')[-1]
            patients.append(folder.removesuffix("_vis"))
    return sorted(set(patients))

                
# the tie-break rule resolver
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
def extract_datetime(filename: str) -> datetime | None:
    match = DATE_PATTERN.search(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def tie_break_resolver(filenames: list[str]) -> str:
    """Tie breaker rule for patients with multiple .tif images"""
    dated = [(extract_datetime(f), f) for f in filenames]
    dated = [(dt, f) for dt, f in dated if dt is not None]

    if not dated:
        return filenames[0]
    
    dated.sort(key=lambda pair: pair[0]) # sort datetimes ascending
    return dated[-1][1] # returns latest date, the file itself


def cohort_discovery(raw_bucket=RAW_DATA_BUCKET, raw_prefix=RAW_DATA_PREFIX, 
                     processed_bucket=PROCESSED_VISIUM_BUCKET, processed_prefix=PROCESSED_VISIUM_PREFIX,
                     out_dir=RESULTS):
    """
    Cohort patient name discovery
    """
    # building the H&E image inventory
    hne_files = list_raw_hne(raw_bucket, raw_prefix)
    hne_map = defaultdict(list)
    unmatched_hne_files = []

    for key in hne_files:
        filename = key.split('/')[-1]
        match = PATIENT_ID_PATTERN.search(filename)
        if match:
            hne_map[match.group()].append(filename)
        else:
            unmatched_hne_files.append(filename)     # filenames with no recognizable patient ID

    resolved_hne_map = {}
    for pid, filenames in hne_map.items():
        if len(filenames) > 1:
            resolved_hne_map[pid] = tie_break_resolver(filenames)    # pick the latest H&E tif for each patient
        else:
            resolved_hne_map[pid] = filenames[0]    

    # building the Visium inventory
    visium_patients = list_processed_visium(processed_bucket, processed_prefix)

    # unifying these two inventories
    report_rows = []
    all_ids = sorted(set(visium_patients) | set(resolved_hne_map.keys()))

    for pid in all_ids:
        has_tif = pid in resolved_hne_map
        n_tifs = 1 if has_tif else 0
        report_rows.append({
            "patient_id": pid,
            "in_visium": pid in visium_patients,
            "n_hne_tifs_found": n_tifs,
            "in_hne": resolved_hne_map.get(pid, ""),
            "status": (
                "MATCHED" if pid in visium_patients and n_tifs == 1 else
                "MISSING_TIFF" if pid in visium_patients and n_tifs == 0 else
                "TIFF_HAS_NO_VISIUM_MATCH"  # tif exist but no processed visium folder
            )
        })

    final_ids = [row["patient_id"] for row in report_rows if row["status"] == "MATCHED"]
    final_tif_map = {pid: resolved_hne_map[pid] for pid in final_ids}

    # human-readable report for the cohort inventory
    out_dir = Path(out_dir)
    report = pd.DataFrame(report_rows).sort_values(["status", "patient_id"])
    report.to_csv(out_dir/"cohort_inventory_report.csv", index=False)
    print(report['status'].value_counts())
    print(f"\n{len(unmatched_hne_files)} hne filenames had no recognizable patient IDs")
    for f in unmatched_hne_files:
        print(" ", f)

    # machine-readable manifest (to import into src/core/paths.py)
    manifest = {"patient_ids": final_ids, "tif_map": final_tif_map}
    with open(out_dir / "cohort_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


    return final_ids, final_tif_map

if __name__ == "__main__":
    cohort_discovery()