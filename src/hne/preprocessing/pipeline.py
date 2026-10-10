"""
src/hne/preprocessing/pipeline.py
Main preprocessing pipeline - reusable for both single patient and cohort
"""

import logging

from hne.core.paths import PATIENTS
from hne.core.data_io import *
from hne.core.data_io import load_he_slide, slide_um_per_px
from hne.preprocessing.tumor_purity import *
from hne.preprocessing.tiling import crop_and_save_tiles, drop_tiles_outside_slide
from hne.preprocessing.spot_signatures import compute_signatures
from hne.preprocessing.aggregation import aggregate_signatures, binary_scores
from hne.preprocessing_qc.plots import *
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)

def preprocess_patient(patient_id, 
                       mode='single_patient',           # or 'cohort'
                       cfg=PREPROCESSING_CONFIG,
                       qc_tracker=None,
                       verbose=True,                    # console output level (True=INFO, False=WARNING)
                       run_qc_plots=False               # per patient
                       ):       
    """
    Preprocess patients and return metadata - reusable function

    Args:
        patient_id: Patient IDs
        mode: 'single_patient' or 'cohort'

    Returns:
        metadata: dict with all QC info
        tiles_sig: DataFrame or None if failed
    """
    logger.info(f"Starting preprocessing patient: {patient_id}")

    # load patient paths
    paths = PATIENTS[patient_id]

    patient_metadata = {"patient_id": patient_id}
    
    # load data
    # only the layer the pipeline scores from is kept; the dense SCTransform layer alone is ~0.6 GB
    vis = load_visium(paths, keep_layers=["log_norm_count"])
    spots = load_spots(paths)
    
    
    # compute tumor fraction
    merged, meta = attach_tumor_fraction(spots, vis, patient_id, qc_tracker, cfg)
    patient_metadata.update(meta)
    # check to see if the deconvolution column exist and merged df of spots and tumor fractions produced or not
    if merged is None:
        return patient_metadata, None, None, None

    # open the full-resolution H&E scan: spot coordinates are its pixels, and its pixel size
    # sets the tile size. The context manager deletes the temp file on exit.
    with load_he_slide(paths, qc_tracker) as slide:
        if slide is None:
            return patient_metadata, None, None, None

        fullres_pixel_size = slide_um_per_px(slide, patient_id)
        slide_w, slide_h = slide.dimensions
        patient_metadata.update({"slide_width": slide_w, "slide_height": slide_h})

        df, meta, tile_size_px = add_tile_coordinates(merged, fullres_pixel_size, cfg)
        patient_metadata.update(meta)

        final_df, meta = compute_tile_purity(df, patient_id, qc_tracker, cfg)
        patient_metadata.update(meta)

        tumor_tiles_df, meta = filter_tumor_tiles(final_df, patient_id, qc_tracker, cfg)
        patient_metadata.update(meta)

        # check if we have tiles BEFORE proceeding
        if not meta.get('has_tumor_tiles', False):
            logger.warning(f"Skipping remaining steps for {patient_id} no tumor tiles!")
            return patient_metadata, None, None, None

        # tiles the scan does not fully cover have no image to learn from
        tumor_tiles_df, meta = drop_tiles_outside_slide(tumor_tiles_df, tile_size_px, slide_w, slide_h, patient_id)
        patient_metadata.update(meta)
        if not meta["has_tumor_tiles"]:
            logger.warning(f"Skipping remaining steps for {patient_id}: no tumor tile inside the H&E scan")
            return patient_metadata, None, None, None

        # crop and save image tiles directly via OpenSlide
        _, meta = crop_and_save_tiles(tumor_tiles_df, tile_size_px, slide, patient_id)
        patient_metadata.update(meta)
    
    # compute signatures per spot, aggregate per tile
    sig_cols, signature_genes, spots_df, meta = compute_signatures(vis, final_df, patient_id, qc_tracker, cfg)
    patient_metadata.update(meta)
    tiles_sig, meta = aggregate_signatures(spots_df, sig_cols, tile_size_px, tumor_tiles_df, cfg)

    # check patients with zero tiles
    if tiles_sig is None or len(tiles_sig) == 0:
        logger.warning(f"No tumor tiles generated after aggregation for {patient_id} - skipping binarization.")
        return patient_metadata, None, spots_df, sig_cols
    
    tiles_sig.insert(0, "patient_id", patient_id)
    patient_metadata.update(meta)

    tiles_sig_tumor = binary_scores(sig_cols, tiles_sig, cfg)
    save_tile_features(tiles_sig_tumor, patient_id, mode)
    
    # QC plots - separate flag
    if run_qc_plots:
        tumor_spots = spots_df[spots_df["tile_id"].isin(tumor_tiles_df["tile_id"])]
        signature_variation(tumor_spots, sig_cols, patient_id, mode)
        signature_distribution(sig_cols, tumor_spots, patient_id, mode)
        signature_sparsity(sig_cols, tumor_spots, patient_id, mode)
        signature_consistency(vis, tumor_spots, signature_genes, patient_id, mode)
        signature_correlation(sig_cols, tumor_spots, patient_id, mode)
    
    logger.info(f"Completed preprocessing for {patient_id}.")

    return patient_metadata, tiles_sig_tumor, spots_df, sig_cols

