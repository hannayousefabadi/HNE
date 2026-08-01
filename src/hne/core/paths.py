"""src/core/paths.py"""
import json

from hne.core.config import (ROOT, RESULTS, PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX, 
                             RAW_DATA_BUCKET, RAW_DATA_PREFIX)

# s3 paths
class PatientS3Paths:
    """
    S3 paths for patient data
    Handles both processed data (folders) and raw images (files).
    """
    def __init__(self, patient_id: str):
        self.patient_id = patient_id

        # clean patient_id:
        self.clean_id = patient_id.replace('_vis', '')

        self.processed_base = f"s3://{PROCESSED_VISIUM_BUCKET}/{PROCESSED_VISIUM_PREFIX}"
        self.visium_st = f"{self.processed_base}/v2/without_spotclean/stLearn/{patient_id}_vis"
        self.visium_info = f"{self.processed_base}/v2/spaceranger_count/{patient_id}_vis/outs/spatial"

        self.raw_base = f"s3://{RAW_DATA_BUCKET}/{RAW_DATA_PREFIX}"
        self.raw_image_prefix = f"{self.raw_base}/spatial_transcriptomics/Visium/image_files"
        

TILES = ROOT / "tiles"
TILES_SIGNATURE_MATRIX = ROOT / "tile_signature_matrix"
QC_REPORTS = ROOT / "qc_reports"
PREPROCESSING_QC_REPORTS = ROOT / "qc_reports" / "preprocessing_qc"

PREPROCESSED_COHORT = PREPROCESSING_QC_REPORTS / "cohort"
PREPROCESSED_SINGLE_PATIENT = PREPROCESSING_QC_REPORTS / "single_patient"

TILES_FEATURES = ROOT / "feature_sets"
PHIKON_FEATURES = TILES_FEATURES / "phikon_v2_features"

# cohort inventory list
try: 
    with open(RESULTS / "cohort_manifest.json") as f:
        _manifest = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError("cohort_manifest.json not found or invalid, run cohort_discovery() first!") from e       

PATIENT_IDS = _manifest["patient_ids"]
TIF_MAP = _manifest["tif_map"]

PATIENTS = {p: PatientS3Paths(p) for p in PATIENT_IDS}

for path in [TILES, TILES_SIGNATURE_MATRIX, QC_REPORTS, PREPROCESSING_QC_REPORTS, 
             PREPROCESSED_SINGLE_PATIENT, PREPROCESSED_COHORT, 
             TILES_FEATURES, PHIKON_FEATURES,
             RESULTS]:
    path.mkdir(parents=True, exist_ok=True)