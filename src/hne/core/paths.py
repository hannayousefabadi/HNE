"""Paths.py"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# project root
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2] 

ROOT = _find_repo_root(Path(__file__))


# s3 configuration
PROCESSED_VISIUM_BUCKET = os.getenv("PROCESSED_VISIUM_BUCKET")
PROCESSED_VISIUM_PREFIX = os.getenv("PROCESSED_VISIUM_PREFIX")

RAW_DATA_BUCKET = os.getenv("RAW_DATA_BUCKET")
RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX")

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
        self.raw_iamge_prefix = f"{self.raw_base}/spatial_transcriptomics/Visium/image_files"
        

# patients ids -> for testing single patient mode only
# the cohort script will discover patients dynamically from S3  
PATIENT_IDS = [
    "CH_L_282a",
    # others
]

PATIENTS = {p: PatientS3Paths(p) for p in PATIENT_IDS}


TILES = ROOT / "tiles"
TILE_FEATURES = ROOT / "tile_features"
QC_REPORTS = ROOT / "qc_reports"
PREPROCESSING_QC_REPORTS = ROOT / "qc_reports" / "preprocessing_qc"
RESULTS = ROOT / "results"


for path in [TILES, TILE_FEATURES, QC_REPORTS, PREPROCESSING_QC_REPORTS]:
    path.mkdir(parents=True, exist_ok=True)