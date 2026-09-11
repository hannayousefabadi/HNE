"""
/src/hne/models/data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

from hne.core.paths import TILES_SIGNATURE_MATRIX


def load_features_and_targets(features_dir: str, 
                              patient_ids: list, 
                              target_cols: list, 
                              filename_suffix: str):
    """
    Building (X, y) training pairs by joining feature vectors and signature scores from the matched tile_id
    loading tile features from vector stored in '{patient_id}_{tile_id}_{foundationmodel}_features.npy'
    loading gene signature scores from 'tiles_signature_matrix_{patient_id}.csv'

    Returns:
        X: (N_tiles, D) feature array
        y: (N_tiles, len(target_cols)) target array
        tile_meta: list of N_tiles(patient_id, tile_id) for traceability, same order as X/y

    """

    features_dir = Path(features_dir)
    X, y, tile_meta = [], [], []

    for patient_id in patient_ids:
        sig_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv"
        if not sig_csv.exists():
            continue
        tiles_df = pd.read_csv(sig_csv)
        if not all (col in tiles_df.columns for col in target_cols):
            continue

        for _, row in tiles_df.iterrows():
            tile_id = row["tile_id"]
            target_vector = row[target_cols].to_numpy(dtype=np.float32)
            if np.isnan(target_vector).any():
                continue # skip the tile if any of the signatures are missing

            npy_path = features_dir / f"{patient_id}_{tile_id}_{filename_suffix}.npy"
            if not npy_path.exists():
                continue

            X.append(np.load(npy_path))
            y.append(target_vector)
            tile_meta.append((patient_id, tile_id))

    if not X:
        return np.empty((0,1024), dtype=np.float32), np.empty(0, len(target_cols), dtype=np.float32), []

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32), tile_meta



