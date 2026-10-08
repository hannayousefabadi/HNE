"""scripts/cohort_inventory/cohort_inventory.py

Builds the cohort manifest: which patients have Space Ranger output and a full-resolution
H&E scan, both from the same version folder of the processed dataset.
"""

import re
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from hne.core.s3_io import S3DataLoader
from hne.core.config import (RESULTS, PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX, PROCESSED_VERSION,
                             RAW_DATA_BUCKET, RAW_DATA_PREFIX)


loader = S3DataLoader()
output_dir = RESULTS / "cohort_metadata"

# CH_L_<digits><optional letter>
PATIENT_ID_PATTERN = re.compile(r"CH_L_\d+[a-z]?")
# a 3000x3000 CytAssist image is at most 30 MB; the smallest genuine scan in this cohort is 48 MB
# (a 17,600 x 14,400 px frame). This only catches a CytAssist image placed in the H&E folder:
# the real check is at open time, where a slide that does not contain the tiles is refused.
MIN_HE_SIZE_BYTES = 35 * 1024 ** 2


def list_fullres_he(bucket, prefix):
    """
    List the full-resolution H&E scans: {version}/converted_he/{patient_id}_vis.tif in the
    processed dataset. These are the images Space Ranger registered the spots to.
    Returns {patient_id: (filename, size_bytes)}.
    """
    full_prefix = f"{prefix.rstrip('/')}/{PROCESSED_VERSION}/converted_he/"
    paginator = loader.s3_client.get_paginator('list_objects_v2')
    files = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter='/'):
        for obj in page.get('Contents', []):
            filename = obj['Key'].split('/')[-1]
            # PatientS3Paths reads the *_vis sample of each patient, so only its scan is matched
            if filename.endswith('_vis.tif'):
                files[filename[:-len('_vis.tif')]] = (filename, obj['Size'])
    return files


def list_cytassist_images(bucket, prefix):
    """
    List the CytAssist instrument images (3000x3000 px, named CAVG...). They are not H&E
    scans and nothing is cropped from them; the manifest keeps them for the registration audit.
    """
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
    full_prefix = f"{prefix.rstrip('/')}/{PROCESSED_VERSION}/spaceranger_count/"
    paginator = loader.s3_client.get_paginator('list_objects_v2')
    patients = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter='/'):
        for common in page.get('CommonPrefixes', []):
            folder = common['Prefix'].rstrip('/').split('/')[-1]
            folder = re.sub(r'_(vis|vbu)$', '', folder)
            patients.append(folder)
    return sorted(set(patients))

                
# the tie-break rule resolver
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
def extract_datetime(filename: str) -> datetime | None:
    match = DATE_PATTERN.search(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")


def tie_break_resolver(filenames: list[str]) -> str:
    """Tie breaker rule for patients with multiple CytAssist images: the latest capture wins"""
    dated = [(extract_datetime(f), f) for f in filenames]
    dated = [(dt, f) for dt, f in dated if dt is not None]

    if not dated:
        return filenames[0]
    
    dated.sort(key=lambda pair: pair[0]) # sort datetimes ascending
    return dated[-1][1] # returns latest date, the file itself


def cohort_discovery(raw_bucket=RAW_DATA_BUCKET, raw_prefix=RAW_DATA_PREFIX, 
                     processed_bucket=PROCESSED_VISIUM_BUCKET, processed_prefix=PROCESSED_VISIUM_PREFIX,
                     out_dir=output_dir):
    """
    Cohort patient name discovery
    """
    # full-resolution H&E scans and Space Ranger output, same version folder
    he_files = list_fullres_he(processed_bucket, processed_prefix)
    visium_patients = list_processed_visium(processed_bucket, processed_prefix)

    # CytAssist images, for the audit only
    cytassist_by_patient = defaultdict(list)
    for key in list_cytassist_images(raw_bucket, raw_prefix):
        filename = key.split('/')[-1]
        match = PATIENT_ID_PATTERN.search(filename)
        if match:
            cytassist_by_patient[match.group()].append(filename)
    cytassist_map = {pid: tie_break_resolver(names) if len(names) > 1 else names[0]
                     for pid, names in cytassist_by_patient.items()}

    # unifying the two inventories
    report_rows = []
    for pid in sorted(set(visium_patients) | set(he_files)):
        filename, size = he_files.get(pid, ("", 0))
        report_rows.append({
            "patient_id": pid,
            "in_visium": pid in visium_patients,
            "he_file": filename,
            "he_size_mb": round(size / 1024 ** 2, 1),
            "cytassist_file": cytassist_map.get(pid, ""),
            "status": (
                "HE_HAS_NO_VISIUM_MATCH" if pid not in visium_patients else   # scan exists but no Space Ranger folder
                "MISSING_HE" if not filename else
                "HE_TOO_SMALL" if size < MIN_HE_SIZE_BYTES else                # not a full-resolution scan
                "MATCHED"
            )
        })

    final_ids = [row["patient_id"] for row in report_rows if row["status"] == "MATCHED"]
    he_map = {pid: he_files[pid][0] for pid in final_ids}

    # human-readable report for the cohort inventory
    out_dir = Path(out_dir)
    report = pd.DataFrame(report_rows).sort_values(["status", "patient_id"])
    report.to_csv(out_dir/"cohort_inventory_report.csv", index=False)
    print(f"Processed dataset version: {PROCESSED_VERSION}")
    print(report['status'].value_counts())
    matched = report[report["status"] == "MATCHED"]
    if not matched.empty:
        print(f"\nH&E scan size (MB): median {matched['he_size_mb'].median():.0f}, "
              f"min {matched['he_size_mb'].min():.0f}, max {matched['he_size_mb'].max():.0f}")

    # machine-readable manifest (imported by src/hne/core/paths.py)
    manifest = {
        "processed_version": PROCESSED_VERSION,
        "patient_ids": final_ids,
        "he_map": he_map,
        "cytassist_map": {pid: cytassist_map[pid] for pid in final_ids if pid in cytassist_map},
    }
    with open(out_dir / "cohort_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return final_ids, he_map

if __name__ == "__main__":
    cohort_discovery()
