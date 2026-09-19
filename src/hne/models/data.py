"""
/src/hne/models/data.py
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from hne.core.paths import TILES_SIGNATURE_MATRIX


def get_cohort_statistics(
    patient_ids: List[str],
    target_cols: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute population mean and std for target columns across all specified patients.
    Used to compute leak-free z-scores using training patients only.
    """
    all_dfs = []
    for pid in patient_ids:
        sig_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{pid}.csv"
        if sig_csv.exists():
            df = pd.read_csv(sig_csv)
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No signature CSVs found for patient list: {patient_ids}")

    combined = pd.concat(all_dfs, ignore_index=True)
    means = {}
    stds = {}
    for col in target_cols:
        # Base column if a '_cohort_z' name was requested
        base_col = col[:-9] if col.endswith("_cohort_z") else col
        if base_col not in combined.columns:
            raise KeyError(f"Column '{base_col}' not found in signature files.")
        means[col] = float(combined[base_col].mean())
        stds[col] = float(combined[base_col].std(ddof=0)) or 1.0

    return means, stds



def load_features_and_targets(
    features_dir: str,
    patient_ids: List[str],
    target_cols: List[str],
    filename_suffix: str,
    cohort_stats: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None,
):
    """
    Building (X, y) training pairs by joining feature vectors and signature scores from the matched tile_id.
    
    Dynamically creates '{col}_cohort_z' columns on the fly without modifying 
    the underlying CSV files on disk.

    Args:
        features_dir: Directory containing .npy feature files
        patient_ids: List of patient IDs to include
        target_cols: List of target columns (supports both raw names and '{name}_cohort_z')
        filename_suffix: Suffix for feature npy files
        cohort_stats: Optional tuple of (means_dict, stds_dict). If provided, scales
                      using these stats; otherwise computes from current patient subset.

    Returns:
        X: (N_tiles, D) feature array
        y: (N_tiles, len(target_cols)) target array
        tile_meta: list of (patient_id, tile_id) tuples matching rows of X and y
    """
    features_dir = Path(features_dir)
    X, y, tile_meta = [], [], []

    # check if any requested columns require cohort-level z-scoring
    needs_scaling = [col for col in target_cols if col.endswith("_cohort_z")]
    if needs_scaling and cohort_stats is None:
        cohort_stats = get_cohort_statistics(patient_ids, needs_scaling)

    means, stds = cohort_stats if cohort_stats else ({}, {})

    for patient_id in patient_ids:
        sig_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv"
        if not sig_csv.exists():
            continue
        tiles_df = pd.read_csv(sig_csv)

        # add cohort z-score columns without modifying the raw values
        for col in target_cols:
            if col.endswith("_cohort_z") and col not in tiles_df.columns:
                base_col = col[:-9]
                if base_col in tiles_df.columns:
                    tiles_df[col] = (tiles_df[base_col] - means[col]) / (stds[col] + 1e-8)

        if not all(col in tiles_df.columns for col in target_cols):
            continue

        for _, row in tiles_df.iterrows():
            tile_id = row["tile_id"]
            target_vector = row[target_cols].to_numpy(dtype=np.float32)
            if np.isnan(target_vector).any():
                continue  # skip tile if any signature is missing

            npy_path = features_dir / f"{patient_id}_{tile_id}_{filename_suffix}.npy"
            if not npy_path.exists():
                continue

            X.append(np.load(npy_path))
            y.append(target_vector)
            tile_meta.append((patient_id, tile_id))

    if not X:
        return np.empty((0, 1024), dtype=np.float32), np.empty((0, len(target_cols)), dtype=np.float32), []

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32), tile_meta

