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
# hard coded list just in case 
PATIENT_IDS = [
    "CH_L_275a",
    "CH_L_276a",
    "CH_L_277a",
    "CH_L_278a",
    "CH_L_279a",
    "CH_L_280a",
    "CH_L_281a",
    "CH_L_281b",
    "CH_L_282a",
    "CH_L_283a",
    "CH_L_284a",
    "CH_L_285a",
    "CH_L_287a",
    "CH_L_288a",
    "CH_L_289a",
    "CH_L_290a",
    "CH_L_291a",
    "CH_L_292a",
    "CH_L_293a",
    "CH_L_294a",
    "CH_L_295a",
    "CH_L_296a",
    "CH_L_298a",
    "CH_L_299a",
    "CH_L_300a",
    "CH_L_301a",
    "CH_L_302a",
    "CH_L_303a",
    "CH_L_305a",
    "CH_L_306a",
    "CH_L_306b",
    "CH_L_307a",
    "CH_L_308a",
    "CH_L_309a",
    "CH_L_310a",
    "CH_L_311a",
    "CH_L_312a",
    "CH_L_313a",
    "CH_L_314a",
    "CH_L_315a",
    "CH_L_317a",
    "CH_L_318a",
    "CH_L_319a",
    "CH_L_320a",
    "CH_L_321a",
    "CH_L_322a",
    "CH_L_323a",
    "CH_L_326a",
    "CH_L_327a",
    "CH_L_328a",
    "CH_L_329a",
    "CH_L_330a",
    "CH_L_333a",
    "CH_L_334a",
    "CH_L_335a",
    "CH_L_336a",
    "CH_L_337a",
    "CH_L_338a",
    "CH_L_339a",
    "CH_L_340a",
    "CH_L_341a",
    "CH_L_342a",
    "CH_L_344a",
    "CH_L_345a",
    "CH_L_346a",
    "CH_L_347a",
    "CH_L_348a",
    "CH_L_349a",
    "CH_L_350a",
    "CH_L_351a",
    "CH_L_352a",
    "CH_L_353a",
    "CH_L_356a",
    "CH_L_359a",
    "CH_L_360a",
    "CH_L_361a",
    "CH_L_362a",
    "CH_L_363a",
    "CH_L_364a",
    "CH_L_367a",
    "CH_L_368a",
    "CH_L_369a",
    "CH_L_371a",
    "CH_L_372a",
    "CH_L_373a",
    "CH_L_375a",
    "CH_L_376a",
    "CH_L_377a",
    "CH_L_382a",
    "CH_L_383a",
    "CH_L_384a",
    "CH_L_385a",
    "CH_L_386a",
    "CH_L_387a",
    "CH_L_388a",
    "CH_L_389a",
    "CH_L_390a",
    "CH_L_391a",
    "CH_L_392a",
    "CH_L_393a",
    "CH_L_394a",
    "CH_L_395a",
    "CH_L_396a",
    "CH_L_397a",
    "CH_L_398a",
    "CH_L_399a",
    "CH_L_400a",
    "CH_L_401a",
    "CH_L_402a",
    "CH_L_403a",
    "CH_L_405a",
    "CH_L_406a",
    "CH_L_407a",
    "CH_L_409a",
    "CH_L_410a",
    "CH_L_411a",
    "CH_L_412a",
    "CH_L_413a",
    "CH_L_414a",
    "CH_L_414b",
    "CH_L_415a",
    "CH_L_416a",
    "CH_L_417a",
    "CH_L_418a",
    "CH_L_419a",
    "CH_L_420a",
    "CH_L_420b",
    "CH_L_421a",
    "CH_L_421b",
    "CH_L_422a",
    "CH_L_423a",
    "CH_L_424a",
    "CH_L_425a",
    "CH_L_426a",
    "CH_L_427a",
    "CH_L_428a",
    "CH_L_429a",
    "CH_L_430a",
    "CH_L_431a",
    "CH_L_432a",
    "CH_L_433a",
    "CH_L_437a",
    "CH_L_437b",
    "CH_L_438a",
    "CH_L_439a",
    "CH_L_439b",
    "CH_L_440a",
    "CH_L_442a"
]



PATIENTS = {p: PatientS3Paths(p) for p in PATIENT_IDS}


TILES = ROOT / "tiles"
TILE_FEATURES = ROOT / "tile_features"
QC_REPORTS = ROOT / "qc_reports"
PREPROCESSING_QC_REPORTS = ROOT / "qc_reports" / "preprocessing_qc"
RESULTS = ROOT / "results"


for path in [TILES, TILE_FEATURES, QC_REPORTS, PREPROCESSING_QC_REPORTS]:
    path.mkdir(parents=True, exist_ok=True)