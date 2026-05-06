
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
                       TILE_SIZE=100,       # in pixels, ≈ 1 mm in hires image
                       k=2, 
                       tumor_threshold=0.3, # tile at least has 30% tumor purity
                       min_spots=40,        # tile at least has 40 spots
                       qc_tracker=None,
                       logger=None,
                       run_qc=True):        # run_qc controls plots, not logger level
    """
    Process single patient - reusable function
    
    Returns:
        metadata: dict with all QC info
        tiles_sig: DataFrame or None if failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    paths = PATIENTS[patient_id]
    patient_metadata = {"patient_id": patient_id}
    
    vis = load_visium(paths)
    spots = load_spots(paths)
    scales = load_scale_factor(paths)
    img = load_he_image(paths)
    
    merged, meta = attach_tumor_fraction(spots, vis, patient_id, logger)
    patient_metadata.update(meta)
    
    df, meta = add_tile_coordinates(scales, TILE_SIZE, merged, logger)
    patient_metadata.update(meta)
    
    final_df, meta = compute_tile_purity(df, k, logger)
    patient_metadata.update(meta)
    
    qc = QCTracker()
    tumor_tiles, meta = filter_tumor_tiles(final_df, tumor_threshold, min_spots, patient_id, logger, qc_tracker=qc)
    patient_metadata.update(meta)
    
    # Check if we should continue - DECISION LOGIC ONLY HERE
    if not meta.get('has_tumor_tiles', False):
        logger.warning(f"Skipping {patient_id} - no tumor tiles!")
        return patient_metadata, None
    
    tumor_tiles, meta = crop_and_save_tiles(tumor_tiles, TILE_SIZE, img, logger)
    patient_metadata.update(meta)
    
    sig_cols, signature_genes, spots_df = compute_signatures(vis, df, logger)
    tiles_sig_tumor = aggregate_signatures(spots_df, sig_cols, TILE_SIZE, tumor_tiles, logger)
    tiles_sig_tumor = zscore_and_binary(sig_cols, tiles_sig_tumor, logger)
    save_tile_features(tiles_sig_tumor, patient_id, logger)
    
    # QC plots - separate flag
    if run_qc:
        tumor_spots = spots_df[spots_df["tile_id"].isin(tumor_tiles["tile_id"])]
        signature_variation(tumor_spots, sig_cols, patient_id, logger)
        signature_distribution(sig_cols, tumor_spots, patient_id, logger)
        signature_sparsity(sig_cols, tumor_spots, patient_id, logger)
        signature_consistency(vis, tumor_spots, signature_genes, patient_id, logger)
        signature_correlation(sig_cols, tumor_spots, patient_id, logger)
    
    logger.info(f"Completed preprocessing for {patient_id}.")

    return patient_metadata, tiles_sig_tumor