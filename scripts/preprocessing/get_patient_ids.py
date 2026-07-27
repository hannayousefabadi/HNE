"""get_patient_ids.py"""

import re
import pandas as pd
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
    full_prefix = f"{prefix.rstrip('/')}/v2/spaceranger_count"
    paginator = loader.s3_client.get_paginator('list_objects_v2')
    patients = []
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix, Delimiter='/'):
        for common in page.get('CommonPrefixes', []):
            folder = common['Prefix'].rstrip('/').split('/')[-1]
            patients.append(folder.replace('_vis', ''))
    return sorted(set(patients))

# building the H&E image inventory
hne_files = list_raw_hne(RAW_DATA_BUCKET, RAW_DATA_PREFIX)
hne_map = defaultdict(list)
unmatched_hne_files = []

for key in hne_files:
    filename = key.split('/')[-1]
    match = PATIENT_ID_PATTERN.search(filename)
    if match:
        hne_map[match.group()].append(filename)
    else:
        unmatched_hne_files.append(filename)     # filenames with no recognizable patient ID

# building the Visium inventory
visium_patients = list_processed_visium(PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX)

# comparing these two lists
rows = []
all_ids = sorted(set(visium_patients) | set(hne_map.keys()))

for pid in all_ids:
    n_tifs = len(hne_map.get(pid, []))
    rows.append({
        "patient_id": pid,
        "in_visium": pid in visium_patients,
        "n_hne_tifs_found": n_tifs,
        "in_hne": "; ".join(hne_map.get(pid, [])),
        "status": (
            "MATCHED_1TO1" if pid in visium_patients and n_tifs == 1 else
            "MISSING_TIFF" if pid in visium_patients and n_tifs == 0 else
            "MULTIPLE_TIFFS" if pid in visium_patients and n_tifs > 1 else
            "TIFF_HAS_NO_VISIUM_MATCH"  # tif exist but no processed visium folder
        )
    })

out_dir = Path(RESULTS)
report = pd.DataFrame(rows).sort_values(["status", "patient_id"])
report.to_csv(out_dir/"patient_ids_report.csv", index=False)

print(report['status'].value_counts())
print(f"\n{len(unmatched_hne_files)} hne filenames had no recognizable patient IDs")
for f in unmatched_hne_files:
    print(" ", f)


