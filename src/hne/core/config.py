"""src/core/config.py"""
import os
from pathlib import Path
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
# version folder of the MOSAIC Visium pipeline output. The H&E image, the Space Ranger
# coordinates and the h5ad must all come from the same one.
PROCESSED_VERSION = "v2"

RAW_DATA_BUCKET = os.getenv("RAW_DATA_BUCKET")
RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX")

RESULTS = ROOT / "results"
MODEL_RESULTS = RESULTS / "models"

for path in [RESULTS, MODEL_RESULTS]:
    path.mkdir(parents=True, exist_ok=True)
