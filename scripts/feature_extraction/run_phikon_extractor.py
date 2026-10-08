
"""scripts/feature_extraction/run_phikon_extractor.py"""
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch

from hne.feature_extraction.phikon_v2_model import PhikonV2Extractor
from hne.feature_extraction.patching import require_fullres_slide
from hne.core.paths import (PATIENTS, PATIENT_IDS, TILES_SIGNATURE_MATRIX, 
                            PREPROCESSING_QC_REPORTS, PHIKON_FEATURES, FEATURE_EXTRACTION_QC_REPORTS)
from hne.core.data_io import load_he_slide


def append_rows(rows: list[dict], path: Path):
    """Append records to a CSV log, writing the header only once."""
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def extract_features():
    phikon = PhikonV2Extractor()
    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")
    log_dir = FEATURE_EXTRACTION_QC_REPORTS / "phikon_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    
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

        # 1) extract tiles within slide context
        with load_he_slide(paths) as slide:
            if slide is None:
                continue

            # the slide must be the full-resolution H&E: tile coordinates are used as they are
            slide_w, slide_h = slide.dimensions
            require_fullres_slide(patient_id, slide_w, slide_h, patient_tiles)

            tile_records = phikon.extract_patient_tiles(
                patient_id=patient_id,
                slide=slide,
                tiles_df=patient_tiles,
                fullres_pixel_size=fullres_px_size,
                tile_size_px_fullres=tile_size_px,
                output_dir=PHIKON_FEATURES,
                batch_size=16,
            )

        # 2) extraction log: what was read for this patient, and from where
        tile_log = pd.DataFrame(tile_records)
        n_oob = int(tile_log["n_out_of_bounds"].sum()) if not tile_log.empty else 0
        append_rows(tile_records, log_dir / "tile_log.csv")
        append_rows([{
            "patient_id": patient_id,
            "slide_width": slide_w,
            "slide_height": slide_h,
            "um_per_px": fullres_px_size,
            "tile_size_px": tile_size_px,
            "n_tiles_processed": len(tile_log),
            "n_tiles_written": int(tile_log["embedding_written"].sum()) if not tile_log.empty else 0,
            "n_tiles_with_out_of_bounds": int((tile_log["n_out_of_bounds"] > 0).sum()) if not tile_log.empty else 0,
            "n_patches_out_of_bounds": n_oob,
        }], log_dir / "patient_log.csv")
        if n_oob:
            tqdm.write(f"WARNING {patient_id}: {n_oob} patch windows fell outside the "
                       f"{slide_w}x{slide_h} slide and were skipped (see tile_log.csv)")

        # force garbage collection and flush CUDA cache between slides
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
    print("\nFeature extraction with Phikon-v2 completed successfully!")


if __name__ == "__main__":
    extract_features()


