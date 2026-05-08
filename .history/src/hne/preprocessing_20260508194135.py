"""Main preprocessing pipeline - reusable for both single patient and cohort"""

import logging
from hne.paths import PATIENTS
from hne.data_io import *
from hne.tumor_purity import *
from hne.tiling import crop_and_save_tiles
from hne.spot_signatures import compute_signatures
from hne.aggregation import aggregate_signatures, zscore_and_binary
from hne.spots_qc import *


def preprocess_patient(patient_id, 
                       mode='single_patient', # or 'cohort'
                       tile_size=100,        # in pixels, ≈ 1 mm in hires image
                       k=2, 
                       tumor_threshold=0.3,  # tile at least has 30% tumor purity
                       min_spots=40,         # tile at least has 40 spots
                       qc_tracker=None,
                       verbose=True,          # console output level (True=INFO, False=WARNING)
                       run_qc_plots=True
                       ):       
    """
    Preprocess patients and return metadata - reusable function
    Args:
    
    Returns:
        metadata: dict with all QC info
        tiles_sig: DataFrame or None if failed
    """
    # set logger level based on verbose
    if verbose:
        logger.setLevel(logging.INFO)
        logger.info(f"Starting preprocessing patient: {patient_id}")
    else:
        logger.setLevel(logging.WARNING)         

    # load patient paths
    paths = PATIENTS[patient_id]
    patient_metadata = {"patient_id": patient_id}
    
    # load data
    vis = load_visium(paths)
    spots = load_spots(paths)
    scales = load_scale_factor(paths)
    img = load_he_image(paths)
    
    # compute tumor fraction and tile coords
    merged, meta = attach_tumor_fraction(spots, vis, patient_id, logger)
    patient_metadata.update(meta)

    df, meta = add_tile_coordinates(scales, tile_size, merged, logger)
    patient_metadata.update(meta)
    
    final_df, meta = compute_tile_purity(df, k, logger)
    patient_metadata.update(meta)
    
    # filter tumor tiles (qc_tracker will record it)
    tumor_tiles, meta = filter_tumor_tiles(final_df, tumor_threshold, min_spots, patient_id, logger, qc_tracker)
    patient_metadata.update(meta)
    
    # check if we have tiles BEFORE proceeding
    if not meta.get('has_tumor_tiles', False):
        logger.warning(f"Skipping remaining steps for {patient_id} no tumor tiles!")
        return patient_metadata, None
    
    # crop and save image tiles
    tumor_tiles, meta = crop_and_save_tiles(tumor_tiles, tile_size, img, logger)
    patient_metadata.update(meta)
    
    # compute signatures per spot, aggregate per tile
    sig_cols, signature_genes, spots_df = compute_signatures(vis, df, logger)
    tiles_sig_tumor = aggregate_signatures(spots_df, sig_cols, tile_size, tumor_tiles, logger)
    tiles_sig_tumor = zscore_and_binary(sig_cols, tiles_sig_tumor, logger)
    save_tile_features(tiles_sig_tumor, patient_id, logger)
    
    # QC plots - separate flag
    if run_qc_plots:
        tumor_spots = spots_df[spots_df["tile_id"].isin(tumor_tiles["tile_id"])]
        signature_variation(tumor_spots, sig_cols, patient_id, mode)
        signature_distribution(sig_cols, tumor_spots, patient_id, mode)
        signature_sparsity(sig_cols, tumor_spots, patient_id, mode)
        signature_consistency(vis, tumor_spots, signature_genes, patient_id, mode)
        signature_correlation(sig_cols, tumor_spots, patient_id, mode)
    
    logger.info(f"Completed preprocessing for {patient_id}.")

    return patient_metadata, tiles_sig_tumor

