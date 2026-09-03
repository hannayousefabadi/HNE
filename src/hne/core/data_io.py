"""src/core/data_io"""

import pandas as pd
from pathlib import Path

from hne.core.paths import (PatientS3Paths, TIF_MAP, TILES_SIGNATURE_MATRIX)
from hne.core.s3_io import S3DataLoader

_s3_loader = None

def get_s3_loader():
    global _s3_loader
    if _s3_loader is None:
        _s3_loader = S3DataLoader()
    return _s3_loader        

def load_visium(patient_paths: PatientS3Paths):
    loader = get_s3_loader()
    h5ad_path = f"{patient_paths.visium_st}/{patient_paths.clean_id}_vis_c2l_annots.h5ad"
    return loader.read_h5ad(h5ad_path)

def load_spots(patient_paths: PatientS3Paths):
    loader = get_s3_loader()
    positions_path = f"{patient_paths.visium_info}/tissue_positions.csv"
    return loader.read_csv(positions_path)

def load_scale_factor(patient_paths: PatientS3Paths):
    loader = get_s3_loader()
    scale_path = f"{patient_paths.visium_info}/scalefactors_json.json"
    return loader.read_json(scale_path)

def load_he_image(patient_paths: PatientS3Paths, qc_tracker=None):
    loader = get_s3_loader()

    filename = TIF_MAP.get(patient_paths.clean_id)

    if filename:
        tif_path = f"{patient_paths.raw_image_prefix}/{filename}"
        return loader.read_tif(tif_path)
    elif qc_tracker:
        qc_tracker.add_record(patient_paths.patient_id,
                                  "fullresimg_load",
                                  "EXCLUDE",
                                  f"No full resolution image found found for {patient_paths.patient_id}",
                                  metadata={})    

def load_he_slide(patient_paths: PatientS3Paths, qc_tracker=None):
    """Load H&E image with openslide.OpenSlide for patch extraction."""
    loader = get_s3_loader()
    filename = TIF_MAP.get(patient_paths.clean_id)

    if filename:
        tif_path = f"{patient_paths.raw_image_prefix}/{filename}"
        return loader.read_tif_as_openslide(tif_path)
    elif qc_tracker:
        qc_tracker.add_record(patient_paths.patient_id, "fullresimg_load",
                              "EXCLUDE", f"No full resolution image found for {patient_paths.patient_id}",
                              metadata={})
        
    return None, None   


def save_tile_features(tiles_sig_tumor, patient_id=None, mode='cohort'):
    """
    Save tile‑level signature matrix
    Handle both single_patient DataFrame and a list of DataFrames (cohort) 
    """
    output_dir = Path(TILES_SIGNATURE_MATRIX)
    output_dir.mkdir(parents=True, exist_ok=True)

    # handle list of DataFrames 
    if isinstance(tiles_sig_tumor, list):
        df = pd.concat(tiles_sig_tumor, ignore_index=True)
        file_name = f"tiles_signature_matrix_{mode}.csv"
    # handle single patient
    else:
        df = tiles_sig_tumor
        file_name = f"tiles_signature_matrix_{patient_id}.csv"

    df.to_csv(output_dir / file_name, index=False)


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




