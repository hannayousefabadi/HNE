"""
Tumor purity preprocessing.

QC thresholds:
- Mean tumor fraction of tiles < 0.2       -> EXCLUDE
- Missing tumor fraction > 10%             -> FLAG
- Mean tile purity < 0.30                  -> FLAG
- 0 tumor tiles after filtering            -> EXCLUDE
- <20 tumor tiles after filtering          -> FLAG
"""

import pandas as pd
from hne.utils import get_logger

logger = get_logger()

def attach_tumor_fraction(spots, 
                          vis, 
                          patient_id, 
                          qc_tracker=None):
    """
    Compute tumor fraction per spot
    Returns:
        merged_df
        metadata_dict
    """
    MEAN_TUMOR_FRACTION_THRESHOLD = 0.2

    in_tissue_spots = spots[spots["in_tissue"] == 1].copy()
    in_tissue_spots["barcode"] = in_tissue_spots["barcode"].astype(str)

    grp = vis.obsm["deconv_sample_level_custom1_frac"]
    tumor_fraction = grp[[col for col in grp.columns if "Tu_" in col]].sum(axis=1)  # e.g. Tu_CH_L_282a_c01 & Tu_CH_L_282a_nos for patient "CH_L_282a" 
    tumor_df = tumor_fraction.to_frame(name="tumor_fraction")
    tumor_df.index = tumor_df.index.astype(str)

    merged = in_tissue_spots.merge(tumor_df, how="left", left_on="barcode", right_index=True)

    # metadata
    metadata = {
        "n_spots_in_tissue": len(merged),
        "n_spots_missin_tumor_fraction": int(merged['tumor_fraction'].isna().sum()),
        "mean_tumor_fraction": float(merged['tumor_fraction'].mean()),
        "median_tumor_fraction": float(merged['tumor_fraction'].median())
    }

    logger.info(f"Attached tumor fraction to {metadata['n_spots_in_tissue']} "
                "tissue spots")
    
    logger.debug(
                 f"Tumor fraction stats | "
                 f"missing={metadata["n_spots_missin_tumor_fraction"]}"
                 f"mean={metadata["mean_tumor_fraction"]:.2f}"
                 f"median={metadata["median_tumor_fraction"]:.2f}"
                 )
    
    if metadata['n_spots_missin_tumor_fraction'] > 0:
        logger.debug(f"{metadata['n_spots_missin_tumor_fraction']} spots missing tumor fraction")
    

    if qc_tracker:

        if metadata["n_spots_in_tissue"] == 0:
            qc_tracker.add_record(patient_id,
                                  "tumor_fraction",
                                  "EXCLUDE",
                                  f"No in-tissue spots found",
                                  metadata)    
 
        elif (metadata["n_spots_missin_tumor_fraction"] / metadata["n_spots_in_tissue"]) > 0.1:
            missing_fraction = (metadata["n_spots_missin_tumor_fraction"] / metadata["n_spots_in_tissue"])
            qc_tracker.add_record(patient_id, 
                                  "tumor_fraction",
                                  "FLAG",
                                  f"{missing_fraction * 100:.1f}% missing tumor fraction",
                                  metadata)
        
        elif metadata["mean_tumor_fraction"] < MEAN_TUMOR_FRACTION_THRESHOLD:
            qc_tracker.add_record(patient_id, "tumor_fraction", "EXCLUDE",
                                  f"Mean tumor fraction is {metadata['mean_tumor_fraction']}",
                                  metadata)
            
        else:
            qc_tracker.add_record(patient_id, "tumor_fraction", "OK",
                                  "Tumor fraction attached successfully", 
                                  metadata)
        
    return merged, metadata


def add_tile_coordinates(scales, 
                         tile_size,     # ≈ 1 mm in hires image
                         merged,
                         patient_id=None
                         ):
    """
    Derive tile coordinates from spot positions
    Returns:
        df
        metadata
    """
    MIN_INITIAL_TILES = 100

    spot_diameter = 55  # Visium spot diameter in µm

    spot_diameter_fullres = scales["spot_diameter_fullres"]
    tissue_hires_scalef = scales["tissue_hires_scalef"]

    fullres_pixel_size = spot_diameter / spot_diameter_fullres
    hires_pixel_size = spot_diameter / (spot_diameter_fullres * tissue_hires_scalef)

    logger.debug(f"Pixel sizes - Fullres: {fullres_pixel_size:.2f} µm/pixel, Hires: {hires_pixel_size:.2f} µm/pixel")

    merged["x_hires"] = merged["pxl_col_in_fullres"] * tissue_hires_scalef
    merged["y_hires"] = merged["pxl_row_in_fullres"] * tissue_hires_scalef
    merged["tile_col"] = (merged["x_hires"] // tile_size).astype(int)    # tile_col = index tiles (0,1,2,3,…) vertically
    merged["tile_row"] = (merged["y_hires"] // tile_size).astype(int)    # tile_row = index tiles (0,1,2,3,…) horizontally
    merged["tile_id"] = merged["tile_row"].astype(str) + "-" + merged["tile_col"].astype(str)

    metadata = {
        "n_initial_tiles": len(merged['tile_id'].unique()),
        "fullres_pixel_size": fullres_pixel_size,
        "hires_pixel_size": hires_pixel_size
    }
    
    logger.info(f"Created {metadata['n_initial_tiles']} initial tiles")

    # warning if very few tiles
    if metadata['n_initial_tiles'] < MIN_INITIAL_TILES:
        msg = f"Very few initial tiles created: {metadata['n_initial_tiles']}"
        logger.warning(msg)

    return merged, metadata


def compute_tile_purity(
        df: pd.DataFrame,
        k=2,
        patient_id=None,
        qc_tracker=None
        ) -> pd.DataFrame:
    """
    Computes Bayesian tumor purity per tile.

    Returns:
        final_df
        metadata
    """
    MEAN_PURITY_THRESHOLD = 0.3

    grouped = df.groupby("tile_id")["tumor_fraction"]
    alpha = grouped.sum()
    n_spots = grouped.count()

    tile_purity = (alpha + 0.5 * k) / (n_spots + k)     # prior mean = 0.5 (agnostic)
    tile_purity = tile_purity.rename("tile_purity")

    final_df = df.merge(tile_purity, on="tile_id", how="left")

    metadata = {"k_value": k,
                "mean_tile_purity": float(tile_purity.mean()),
                "median_tile_purity": float(tile_purity.median()),
                "tile_purity_std": float(tile_purity.std())
                }
    
    msg = f"Bayesian tile purtiy computed (k={k}): mean={metadata['mean_tile_purity']:.3f}"
    logger.info(msg)

    # warning is tumor purity is very low
    if metadata['mean_tile_purity'] < MEAN_PURITY_THRESHOLD:
        msg = f"Mean tile purity is very low: {metadata['mean_tile_purity']}"
        logger.warning(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tile_purity", "FLAG", msg, metadata)
    else:
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tile_purity", "OK", 
                                  f"Mean tile purity is above {MEAN_PURITY_THRESHOLD * 100}%",
                                  metadata)    

    return final_df, metadata


def filter_tumor_tiles(df, 
                       tumor_threshold,
                       min_spots,        
                       patient_id,
                       qc_tracker=None
                       ):
    """
    Filter tumor tiles (purity + min_spots)
    Returns:
        tumor_tiles
        metadata
    """
    MIN_FINAL_TILES = 20


    tiles_stats = df.groupby(["tile_row", "tile_col"]).agg(
        tile_id=("tile_id", "first"),
        tile_purity=("tile_purity", "first"),
        n_spots=("barcode", "count")
    ).reset_index()

    tiles_stats["n_spots"].describe()

    tumor_tiles = tiles_stats.query(
    "tile_purity >= @tumor_threshold and n_spots >= @min_spots"
    )

    metadata = {
        "tiles_before_filter": len(tiles_stats),
        "n_tumor_tiles": len(tumor_tiles),
        "has_tumor_tiles": len(tumor_tiles) > 0, 
        "tumor_threshold": tumor_threshold,
        "min_spots": min_spots
    }

    logger.info(f"Filtered from {metadata['tiles_before_filter']} to {metadata['n_tumor_tiles']} tumor tiles")

    # add QC records
    if metadata['n_tumor_tiles'] == 0:
        msg = f"No tumor tiles found! Check threshold={tumor_threshold}, min_spots={min_spots}"
        logger.error(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "EXCLUDE", msg, metadata)
        
    elif metadata['n_tumor_tiles'] < MIN_FINAL_TILES:
        msg = f"Only {len(tumor_tiles)} tumor tiles - insufficient for MIL training"
        logger.warning(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "FLAG", msg, metadata)

    else:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "OK", 
                                 f"Found {metadata['n_tumor_tiles']} tumor tiles", metadata)

    return tumor_tiles, metadata






