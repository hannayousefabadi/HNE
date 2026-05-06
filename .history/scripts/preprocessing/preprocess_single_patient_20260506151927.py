import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),                        # to print in console
        # logging.FileHandler("preprcessing_single_patient.log")  # to save to file
    ]
)
logger = logging.getLogger(__name__)


from hne.paths import PATIENTS
from hne.data_io import (load_visium, load_spots, 
                         load_scale_factor, load_he_image, save_tile_features)
from hne.tumor_purity import (attach_tumor_fraction, add_tile_coordinates, 
                              compute_tile_purity, filter_tumor_tiles)
from hne.tiling import crop_and_save_tiles
from hne.spot_signatures import compute_signatures
from hne.aggregation import aggregate_signatures, zscore_and_binary
from hne.spots_qc import (signature_variation, signature_distribution, 
                               signature_sparsity, signature_consistency, signature_correlation)

from hne.spots_qc import QCTracker




def preprocess_patient(patient_id, 
                       TILE_SIZE=100,         # in pixels, ≈ 1 mm in hires image
                       k=2, 
                       tumor_threshold=0.3,   # tile at least has 30% tumor purity
                       min_spots=40,          # tile at least has 40 spots  
                       verbose=True
                       ):
    """Process single patient and return metadata"""

    if verbose:
        logger.info(f"Starting preprocessing for parient: {patient_id}") 
        # all info messages (info, warning, error)  

    else:
        pass
        # only warning and error
     

    paths = PATIENTS[patient_id]
    patient_metadata = {"patient_id": patient_id}

    # 1. Load ST data
    vis = load_visium(paths)
    spots = load_spots(paths)
    scales = load_scale_factor(paths)
    img = load_he_image(paths)


    # 2. Compute/attach tumor fraction per spot
    # 3. Merge spots metadata and tumor fractions
    merged, meta = attach_tumor_fraction(spots, vis, logger, patient_id)
    patient_metadata.update(meta)

    # 4. Derive tile coordinates from spot positions
    df, meta = add_tile_coordinates(scales, TILE_SIZE, merged, logger)
    patient_metadata.update(meta)

    # 5. Compute Bayesian tile purity from spot tumor fractions
    final_df, meta = compute_tile_purity(df, k, logger)
    patient_metadata.update(meta)

    # 6. Filter tumor tiles (purity + min_spots)
    qc = QCTracker()
    tumor_tiles, meta = filter_tumor_tiles(final_df, tumor_threshold, min_spots, patient_id, logger, qc_tracker=qc)
    patient_metadata.update(meta)
    
    if not meta.get('has_tumor_tiles', False):
        logger.warning(f"Skipping remaining steps for {patient_id} (no tumor tiles)")
        return patient_metadata, None    

    # 7. Generate & save image tiles (H&E crops)
    tumor_tiles, meta = crop_and_save_tiles(tumor_tiles, TILE_SIZE, img, logger)
    patient_metadata.update(meta)


    # 8. Compute pathway signatures in spot space
    sig_cols, signature_genes, spots_df = compute_signatures(vis, df, logger)

    # 9. Aggregate spot signatures to tile-level
    tiles_sig_tumor = aggregate_signatures(spots_df, sig_cols, TILE_SIZE, tumor_tiles, logger)

    # 10. Apply z‑score & binary calls
    tiles_sig_tumor = zscore_and_binary(sig_cols, tiles_sig_tumor, logger)

    # 11. Save tile‑level signature matrix (MIL‑ready)
    save_tile_features(tiles_sig_tumor, patient_id, logger)


    # 12. QC metrics (variation, sparsity, consistency, redundancy)
    if verbose:
        tumor_spots = signature_variation(spots_df, tumor_tiles, sig_cols, logger)
        signature_distribution(sig_cols, tumor_spots, logger)
        signature_sparsity(sig_cols, tumor_spots, logger)
        signature_consistency(vis, tumor_spots, signature_genes, logger)
        signature_correlation(sig_cols, tumor_spots, logger)

    logger.info(f"Completed preprocessing for {patient_id}.")

    return patient_metadata, tiles_sig_tumor


if __name__ == "__main__":
    metadata, tiles = preprocess_patient("CH_L_282", verbose=True)
    print(f"\nMetadata summary: {metadata}")


    
    



