import logging

from hne.paths import PATIENTS
from hne.data_io import (load_visium, load_spots, 
                         load_scale_factor, load_he_image, save_tile_features)
from hne.tumor_purity import (attach_tumor_fraction, add_tile_coordinates, 
                              compute_tile_purity, filter_tumor_tiles)
from hne.tiling import crop_and_save_tiles
from hne.spot_signatures import compute_signatures
from hne.aggregation import aggregate_signatures, zscore_and_binary
from hne.signatures_qc import (signature_variation, signature_distribution, 
                               signature_sparsity, signature_consistency, signature_correlation)


TILE_SIZE = 100   

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_patient(patient_id, 
                       TILE_SIZE=100,         # in pixels, ≈ 1 mm in hires image
                       k=2, 
                       tumor_threshold=0.3,   # tile at least has 30% tumor purity
                       min_spots=40           # tile at least has 40 spots  
                       ):
    """Process single patient and return metadata"""

    paths = PATIENTS["CH_L_282"]
    patient_metadata = {"patient_id": patient_id}

    # 1. Load ST data
    vis = load_visium(paths)
    spots = load_spots(paths)
    scales = load_scale_factor(paths)
    img = load_he_image(paths)


    # 2. Compute/attach tumor fraction per spot
    # 3. Merge spots metadata and tumor fractions
    merged, meta = attach_tumor_fraction(spots, vis)
    patient_metadata.update(meta)

    # 4. Derive tile coordinates from spot positions
    df, meta = add_tile_coordinates(scales, TILE_SIZE, merged)
    patient_metadata.update(meta)

    # 5. Compute Bayesian tile purity from spot tumor fractions
    final_df = compute_tile_purity(df, k)

    # 6. Filter tumor tiles (purity + min_spots)
    tumor_tiles = filter_tumor_tiles(final_df,
                      tumor_threshold=0.3, 
                      min_spots=40         
                   )

    # 7. Generate & save image tiles (H&E crops)
    tumor_tiles = crop_and_save_tiles(tumor_tiles, TILE_SIZE, img)

    # 8. Compute pathway signatures in spot space
    sig_cols, signature_genes, spots_df = compute_signatures(vis, df)

    # 9. Aggregate spot signatures to tile-level
    tiles_sig_tumor = aggregate_signatures(spots_df, sig_cols, TILE_SIZE, tumor_tiles)

    # 10. Apply z‑score & binary calls
    tiles_sig_tumor = zscore_and_binary(sig_cols, tiles_sig_tumor).copy()

    # 11. Save tile‑level signature matrix (MIL‑ready)
    save_tile_features(tiles_sig_tumor)

    # 12. QC metrics (variation, sparsity, consistency, redundancy)
    tumor_spots = signature_variation(spots_df, tumor_tiles, sig_cols)
    signature_distribution(sig_cols, tumor_spots)
    signature_sparsity(sig_cols, tumor_spots)
    signature_consistency(vis, tumor_spots, signature_genes)
    signature_correlation(sig_cols, tumor_spots)

    



