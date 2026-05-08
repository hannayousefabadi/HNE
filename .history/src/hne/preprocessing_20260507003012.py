
"""Main preprocessing pipeline - reusable for both single patient and cohort"""

import logging
from hne.paths import PATIENTS
from hne.paths import PREPROCESSING_QC_REPORTS, PREPROCESSING_QC_PLOTS
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
                       verbose=None,
                       save_qc_plots=True):       
    """
    Preprocess patients and return metadata - reusable function
    
    Returns:
        metadata: dict with all QC info
        tiles_sig: DataFrame or None if failed
    """
    if verbose:
        logger.setLevel(logging.INFO)
        logger.info(f"Starting preprocessing patient: {patient_id}")
    else:
        logger.setLevel(logging.WARNING)    

    if save_qc_plots:
        qc_plot_dir = PREPROCESSING_QC_PLOTS / patient_id
        qc_plot_dir.mkdir(parents=True, exist_ok=True)
    else:
       qc_plot_dir = None     

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

    # add QC record
    if qc_tracker:
        if not meta.get('has_tumor_tiles', False):
            qc_tracker.add_record(patient_id, "tumor_filter", "ERROR", 
                                 f"No tumor tiles: threshold={tumor_threshold}, min_spots={min_spots}")
        elif meta['n_tumor_tiles'] < 10:
            qc_tracker.add_record(patient_id, "tumor_filter", "WARNING", 
                                 f"Only {meta['n_tumor_tiles']} tumor tiles")
        else:
            qc_tracker.add_record(patient_id, "tumor_filter", "PASS", 
                                 f"Found {meta['n_tumor_tiles']} tumor tiles")
    
    # Check if we have tiles BEFORE proceeding
    if not meta.get('has_tumor_tiles', False):
        logger.warning(f"Skipping remaining steps for {patient_id} no tumor tiles!")
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