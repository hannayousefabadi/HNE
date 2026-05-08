from pathlib import Path

# project root
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2] 

ROOT = _find_repo_root(Path(__file__))

# patients ids
PATIENT_IDS = [
    "CH_L_282",
    # others
]

# patient-context object
class PatienPaths:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.base = ROOT / "data" / patient_id
        self.visium_st = self.base / "without_spotclean" / "stLearn"
        self.visium_info = self.base / "spaceranger_count" / "spatial" 


PATIENTS = {p: PatienPaths(p) for p in PATIENT_IDS}


TILES = ROOT / "tiles"
TILE_FEATURES = ROOT / "tile_features"
PREPROCESSING_QC_REPORTS = ROOT / "qc_reports" / "preprocessing_qc"
PREPROCESSING_QC_PLOTS = PREPROCESSING_QC_REPORTS / "plots"



# ONE_PATIENT = ROOT / "data" / "CH_L_282" 

# VISIUM_ST = ONE_PATIENT / "without_spotclean" / "stLearn" 
# VISIUM_INFO = ONE_PATIENT / "spaceranger_count" / "spatial" 

