"""Main preprocessing pipeline - reusable for both single patient and cohort"""

import logging
from hne.core.paths import PATIENTS
from hne.core.data_io import *
from hne.preprocessing.tumor_purity import *
from hne.preprocessing.tiling import crop_and_save_tiles
from hne.preprocessing.spot_signatures import compute_signatures
from hne.preprocessing.aggregation import aggregate_signatures, zscore_and_binary
from hne.preprocessing_qc.plots import *

logger = logging.getLogger(__name__)

def preprocess_patient(patient_id, 
                       mode='single_patient',           # or 'cohort'
                       target_physical_size_um=1000,    # <-- 1000 µm = 1 mm
                       k=2, 
                       tumor_threshold=0.3,             # tile at least has 30% tumor purity
                       min_spots=40,                    # tile at least has 40 spots
                       qc_tracker=None,
                       verbose=True,                    # console output level (True=INFO, False=WARNING)
                       run_qc_plots=True,
                       use_s3_discovery=False           # if True, discover patient from S3 dynamically
                       ):       
    """
    Preprocess patients and return metadata - reusable function

    Args:
        patient_id: Patient ID (e.g., "CH_L_282a" or "CH_L_282a_vis")
        mode: 'single_patient' or 'cohort'
        target_physical_size_um: Target tile size in micrometers
        use_s3_discovery: If True, discover patient paths from S3 dynamically

    Returns:
        metadata: dict with all QC info
        tiles_sig: DataFrame or None if failed
    """
    logger.info(f"Starting preprocessing patient: {patient_id}")

    # load patient paths -> supports both hardcoded and dynamic discovery:
    if use_s3_discovery or patient_id not in PATIENTS:
        # dynamic discovery from S3
        paths = PatientS3Paths(patient_id)

    else:
        paths = PATIENTS[patient_id]

    patient_metadata = {"patient_id": patient_id}
    
    # load data
    vis = load_visium(paths)
    spots = load_spots(paths)
    scales = load_scale_factor(paths)
    img = load_he_image(paths, qc_tracker)

    
    # compute tumor fraction and tile coords
    merged, meta = attach_tumor_fraction(spots, vis, patient_id, qc_tracker)
    patient_metadata.update(meta)

    df, meta, tile_size_px = add_tile_coordinates(scales, target_physical_size_um, merged)
    patient_metadata.update(meta)
    
    final_df, meta = compute_tile_purity(df, k, patient_id, qc_tracker)
    patient_metadata.update(meta)
    
    tumor_tiles_df, meta = filter_tumor_tiles(final_df, tumor_threshold, min_spots, patient_id, qc_tracker)
    patient_metadata.update(meta)
    
    # check if we have tiles BEFORE proceeding
    if not meta.get('has_tumor_tiles', False):
        logger.warning(f"Skipping remaining steps for {patient_id} no tumor tiles!")
        return patient_metadata, None, None, None
    
    # crop and save image tiles
    tumor_tiles, meta = crop_and_save_tiles(tumor_tiles_df, tile_size_px, img, patient_id)
    patient_metadata.update(meta)
    
    # compute signatures per spot, aggregate per tile
    sig_cols, signature_genes, spots_df, meta = compute_signatures(vis, final_df, patient_id, qc_tracker)
    patient_metadata.update(meta)
    tiles_sig_tumor, meta = aggregate_signatures(spots_df, sig_cols, tile_size_px, tumor_tiles_df)
    patient_metadata.update(meta)
    tiles_sig_tumor = zscore_and_binary(sig_cols, tiles_sig_tumor)
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

