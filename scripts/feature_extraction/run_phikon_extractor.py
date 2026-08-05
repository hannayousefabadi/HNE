"""scripts/feature_extraction/phikon_extractor.py"""
import pandas as pd
import os
from tqdm import tqdm

from hne.feature_extraction.phikon_v2_model import PhikonV2Extractor
from hne.core.paths import (PATIENTS, PATIENT_IDS, TILES_SIGNATURE_MATRIX, 
                            PREPROCESSING_QC_REPORTS, PHIKON_FEATURES)
from hne.core.data_io import load_he_slide


def extract_features():
    phikon = PhikonV2Extractor()

    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")
    
    for patient_id in tqdm(PATIENT_IDS, desc="Extracting Phikon-v2 features"):
        patient_tile_path = pd.read_csv(TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv")
        if not patient_tile_path:
            print(f"Skipping {patient_id}: no saved tile csv for this patient")

        patient_tiles = pd.read_csv(patient_tile_path)
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
                tiles_df=patient_tiles,
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

