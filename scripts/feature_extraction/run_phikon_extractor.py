
"""scripts/feature_extraction/run_phikon_extractor.py"""
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch

from hne.feature_extraction.phikon_v2_model import PhikonV2Extractor
from hne.core.paths import (PATIENTS, PATIENT_IDS, TILES_SIGNATURE_MATRIX, 
                            PREPROCESSING_QC_REPORTS, PHIKON_FEATURES)
from hne.core.data_io import load_he_slide, get_s3_loader


def extract_features():
    phikon = PhikonV2Extractor()
    loader = get_s3_loader()
    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")
    
    for patient_id in tqdm(PATIENT_IDS, desc="Extracting Phikon-v2 features"):
        tiles_csv_path = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv" 
        if not tiles_csv_path.exists():
            continue

        patient_tiles = pd.read_csv(tiles_csv_path)
        if patient_tiles.empty:
            continue        

        # support resuming        
        existing = list(Path(PHIKON_FEATURES).glob(f"{patient_id}_*_phikon_features.npy"))
        if existing and len(existing) >= len(patient_tiles):
            continue

        patient_meta = metadata[metadata["patient_id"] == patient_id]
        if patient_meta.empty:
            continue

        fullres_px_size = float(patient_meta["fullres_pixel_size"].iloc[0])
        tile_size_px = int(patient_meta["tile_size_pixels"].iloc[0])
        paths = PATIENTS[patient_id]

        # 1) retrieve the registration scale factor for this patient if available
        scale_factor = 1.0
        try:
            scale_json = loader.read_json(f"{paths.visium_info}/scalefactors_json.json")
            if "regist_target_img_scalef" in scale_json:
                scale_factor = float(scale_json["regist_target_img_scalef"])
        except Exception:
            scale_factor = 1.0

        # 2) extract tiles within slide context
        with load_he_slide(paths) as slide:
            if slide is None:
                continue

            # double check if slide is already fullres or downscaled
            slide_w, _ = slide.dimensions
            # if the slide width is already ~3000px, apply the scalefactor
            current_scale = scale_factor if slide_w < 10000 else 1.0

            # scale pixel resolution (µm/px increases as image resolution decreases)
            effective_pixel_size = fullres_px_size / current_scale
            effective_tile_size_px = int(round(tile_size_px * current_scale))    


            phikon.extract_patient_tiles(
                patient_id=patient_id,
                slide=slide,
                tiles_df=patient_tiles,
                fullres_pixel_size=effective_pixel_size,
                tile_size_px_fullres=effective_tile_size_px,
                coord_scale_factor=current_scale,
                output_dir=PHIKON_FEATURES,
                batch_size=16,
            )

        # force garbage collection and flush CUDA cache between slides
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
    print("\nFeature extraction with Phikon-v2 completed successfully!")


if __name__ == "__main__":
    extract_features()


