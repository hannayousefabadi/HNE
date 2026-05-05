from pathlib import Path

# project root
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2] 

ROOT = _find_repo_root(Path(__file__))

# For patient CH_L_282
ONE_PATIENT = ROOT / "data" / "CH_L_282" 

VISIUM_ST = ONE_PATIENT / "without_spotclean" / "stLearn" 
VISIUM_INFO = ONE_PATIENT / "spaceranger_count" / "spatial" 

