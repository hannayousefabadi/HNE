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


def login_huggingface() -> None:
    """
    Authenticate with Hugging Face Hub using credentials from .env, hf_token.txt,
    or environment variables (HF_TOKEN, HUGGING_FACE_HUB_TOKEN).
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = ROOT / "hf_token.txt"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        cred_file = ROOT / "hf_credentials.txt"
        if cred_file.exists():
            # Support either raw token or key=value format
            content = cred_file.read_text().strip()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
                elif line.startswith("token="):
                    token = line.split("=", 1)[1].strip()
                    break
            if not token and content and not any("=" in l for l in content.splitlines()):
                token = content

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


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
