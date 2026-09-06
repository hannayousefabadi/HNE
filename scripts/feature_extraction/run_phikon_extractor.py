"""scripts/feature_extraction/phikon_extractor.py"""
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

from hne.feature_extraction.phikon_v2_model import PhikonV2Extractor
from hne.core.paths import (PATIENTS, PATIENT_IDS, TILES_SIGNATURE_MATRIX, 
                            PREPROCESSING_QC_REPORTS, PHIKON_FEATURES)
from hne.core.data_io import load_he_slide


def extract_features():
    phikon = PhikonV2Extractor()
    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")
    
    for patient_id in tqdm(PATIENT_IDS, desc="Extracting Phikon-v2 features"):
        tiles_csv_path = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv" 
        if not tiles_csv_path.exists():
            print(f"Skipping {patient_id}: no tile signature matrix found (not preprocessed)")
            continue

        # support resuming
        existing = list(Path(PHIKON_FEATURES).glob(f"{patient_id}_*_phikon_features.npy"))
        if existing:
            print(f"Skipping {patient_id}: already extracted ({len(existing)} tile features found)")
            continue
        
        patient_tiles = pd.read_csv(tiles_csv_path)
        if patient_tiles.empty:
            print(f"Skipping {patient_id}: tile csv is empty for this patient")
            continue

        patient_meta = metadata[metadata["patient_id"] == patient_id]
        if patient_meta.empty:
            print(f"Skipping {patient_id}: has no row in cohort metadata")
            continue

        fullres_px_size = patient_meta["fullres_pixel_size"].iloc[0]
        tile_size_px = int(patient_meta["tile_size_pixels"].iloc[0])

        paths = PATIENTS[patient_id]
        slide, tmp_path = load_he_slide(paths)

        if slide is None:
            print(f"Skipping {patient_id}: no slide found")
            continue

        try:
            phikon.extract_patient_tiles(
                patient_id=patient_id,
                slide=slide,
                tiles_df=patient_tiles,      # use the loaded DataFrame
                fullres_pixel_size=fullres_px_size,
                tile_size_px_fullres=tile_size_px,
                output_dir=PHIKON_FEATURES
            )
        finally:
            slide.close()
            os.remove(tmp_path)    

    print("\nFeature extraction with Phikon-v2 compeleted!")


if __name__ == "__main__":
    extract_features()

