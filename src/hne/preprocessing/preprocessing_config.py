"""
src/hne/preprocessing/preprocessing_config.py

single source of truth for preprocessing and its QC parameters
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessingConfig:
    # tile geometry & resolution
    target_physical_size_um: float = 1000.0  # 1 mm tile edge length
    spot_diameter_um: float = 55.0           # standard 10x Visium spot diameter
    min_initial_tiles: int = 30              # minimum tiles before layout warning

    # tumor purity & filtering
    k_prior: float = 2.0                     # bayesian prior pseudo-count weight
    mean_tumor_fraction_threshold: float = 0.2 # patient-level exclusion cutoff (< 20% -> EXCLUDE)
    missing_tumor_fraction_tolerance: float = 0.1 # spot missingness tolerance (> 10% -> FLAG)
    mean_purity_threshold: float = 0.3       # patient-level mean purity warning (< 30% -> FLAG)
    tumor_purity_threshold: float = 0.3      # minimum tile purity required to keep a tile
    min_spots_per_tile: int = 40             # minimum spots required to keep a tile
    min_final_tumor_tiles: int = 10          # patient-level tile yield warning (< 10 -> FLAG)

    # signature scoring & binarization
    ssgsea_processes: int = 4                # multiprocessing cores for ssGSEA
    ssgsea_min_size: int = 1                 # minimum overlap genes to evaluate gene set
    binary_quantile_threshold: float = 0.75  # top quartile cutoff for binary tile calls

    # QC & image filtering
    sparse_threshold: float = 1e-6           # spot score zero cutoff for sparsity plots
    min_tissue_fraction: float = 0.5         # minimum tissue fraction for sub-patches

PREPROCESSING_CONFIG = PreprocessingConfig()

