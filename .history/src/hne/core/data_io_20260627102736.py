import scanpy as sc
import pandas as pd
from PIL import Image
import json
from pathlib import Path

from hne.core.paths import PatienPaths, TILE_FEATURES

def load_visium(paths: PatienPaths):
    h5ad_file = f"{paths.patient_id}a_vis_c2l_annots.h5ad"
    return sc.read_h5ad(paths.visium_st / h5ad_file)


def load_spots(paths: PatienPaths):
    return pd.read_csv(paths.visium_info / "tissue_positions.csv")


def load_scale_factor(paths: PatienPaths):    
    with open(paths.visium_info / "scalefactors_json.json") as f:
        return json.load(f)
    

def load_he_image(paths: PatienPaths):     
    return Image.open(paths.visium_info / "tissue_hires_image.png")


def save_tile_features(tiles_sig_tumor, patient_id=None, mode='cohort'):
    """
    Save tile‑level signature matrix
    Handle both single_patient DataFrame and a list of DataFrames (cohort) 
    """
    output_dir = Path(TILE_FEATURES)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_sig_tumor = tiles_sig_tumor.copy()

    # handle list of DataFrames 
    if isinstance(tiles_sig_tumor, list):
        df = pd.concat(tiles_sig_tumor, ignore_index=True)
        file_name = f"tiles_signature_matrix_{mode}.csv"
    # handle single patient
    else:
        if "patinet_id" not in tiles_sig_tumor.columns:
            tiles_sig_tumor.insert(0, "patient_id", patient_id) # add patinet_id as the first col

        df = tiles_sig_tumor
        file_name = f"tiles_signature_marix_{patient_id}.csv"

    output_path = output_dir / file_name
    df.to_csv(output_path, index=False)


def save_metadata(metadata, output_path):
    """Save metadata dict or list of dicts to csv"""
    
    # handle both a single dict and a list of dicts (single patient vs. cohort)
    if isinstance(metadata, dict):
        metadata = [metadata]

    for item in metadata:
        for key, value in item.items():
            if isinstance(value, set):
                item[key] = sorted(value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, set):
                        value[subkey] = sorted(subvalue)

    metadata_df = pd.DataFrame(metadata)

    # flatten nested dictionaries if any
    for col in metadata_df.columns:
        if metadata_df[col].apply(lambda x: isinstance(x, dict)).any():
            # expand nested dicts into separate columns
            expanded = metadata_df[col].apply(pd.Series)
            expanded = expanded.add_prefix(f"{col}_")
            metadata_df = metadata_df.drop(columns=[col]).join(expanded)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_csv(output_path, index=False)
    
    return metadata_df
