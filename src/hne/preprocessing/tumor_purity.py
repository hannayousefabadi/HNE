"""
src/hne/preprocessing/tumor_purity.py
"""
import pandas as pd
from hne.utils import get_logger
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG

logger = get_logger()

def attach_tumor_fraction(spots, 
                          vis, 
                          patient_id=None, 
                          qc_tracker=None,
                          cfg=PREPROCESSING_CONFIG):
    """
    Compute tumor fraction per spot
    """
    deconv_key = "deconv_sample_level_custom1_frac"

    if deconv_key not in vis.obsm:
        if qc_tracker and patient_id:
            qc_tracker.add_record(patient_id, "tumor_fraction", "EXCLUDE",
                                  f"Missing '{deconv_key} in obsm - no deconvolution available",
                                  metadata={})
            logger.warning(f"{patient_id}: missing {deconv_key} in obsm - skipping")
            return None, {"has_tumor_fraction": False}

    in_tissue_spots = spots[spots["in_tissue"] == 1].copy()
    in_tissue_spots["barcode"] = in_tissue_spots["barcode"].astype(str)

    grp = vis.obsm[deconv_key]
    tumor_fraction = grp[[col for col in grp.columns if "Tu_" in col]].sum(axis=1)  # e.g. Tu_CH_L_282a_c01 & Tu_CH_L_282a_nos for patient "CH_L_282a" 
    tumor_df = tumor_fraction.to_frame(name="tumor_fraction")
    tumor_df.index = tumor_df.index.astype(str)

    merged = in_tissue_spots.merge(tumor_df, how="left", left_on="barcode", right_index=True)

    # metadata
    metadata = {
        "n_spots_in_tissue": len(merged),
        "n_spots_missin_tumor_fraction": int(merged['tumor_fraction'].isna().sum()),
        "mean_tumor_fraction": float(merged['tumor_fraction'].mean())
    }

    logger.info(f"Attached tumor fraction to {metadata['n_spots_in_tissue']} "
                "tissue spots")   

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
        elif metadata["mean_tumor_fraction"] < cfg.mean_tumor_fraction_threshold:
            qc_tracker.add_record(patient_id, "tumor_fraction", "EXCLUDE",
                                  f"Mean tumor fraction is {metadata['mean_tumor_fraction']}",
                                  metadata)
        else:
            qc_tracker.add_record(patient_id, "tumor_fraction", "OK",
                                  "Tumor fraction attached successfully", 
                                  metadata)
        
    return merged, metadata


def add_tile_coordinates(scales,
                         merged,
                         cfg=PREPROCESSING_CONFIG
                         ):
    """
    Derive tile coordinates from spot positions based on slide's physical size
    """
    spot_diameter_fullres = scales["spot_diameter_fullres"]
    fullres_pixel_size = cfg.spot_diameter_um / spot_diameter_fullres
    # dynamic calculation: how many fullres pixels are needed to reach the target physical size
    tile_size_px = int(cfg.target_physical_size_um / fullres_pixel_size)
    
    # drop spots with negative pixel coordinates before tiling
    merged = merged[(merged["pxl_col_in_fullres"] >= 0) & merged["pxl_row_in_fullres"] >= 0].copy()
    # use the dynamically calculated pixel size for the grid
    # hard assignment: using the floor division here to assign every single spot to exactly one tile
    # in fullres coordinates 
    merged["tile_col"] = (merged["pxl_col_in_fullres"] // tile_size_px).astype(int)    # tile_col = index tiles (0,1,2,3,…) horizontally -> X, image width 
    merged["tile_row"] = (merged["pxl_row_in_fullres"] // tile_size_px).astype(int)    # tile_row = index tiles (0,1,2,3,…) vertically -> Y, image height
    merged["tile_id"] = merged["tile_row"].astype(str) + "-" + merged["tile_col"].astype(str)

    metadata = {
        "tile_id": sorted(merged["tile_id"].unique().tolist()),
        "n_initial_tiles": len(merged['tile_id'].unique()),
        "fullres_pixel_size": fullres_pixel_size,
        "tile_size_pixels": tile_size_px
    }
    
    logger.info(f"Created {metadata['n_initial_tiles']} initial tiles")

    # warning if very few tiles
    if metadata['n_initial_tiles'] < cfg.min_initial_tiles:
        msg = f"Very few initial tiles created: {metadata['n_initial_tiles']}"
        logger.warning(msg)

    return merged, metadata, tile_size_px


def compute_tile_purity(
        df: pd.DataFrame,
        patient_id=None,
        qc_tracker=None,
        cfg=PREPROCESSING_CONFIG
        ) -> pd.DataFrame:
    """
    Computes Bayesian tumor purity per tile.
    """
    grouped = df.groupby("tile_id")["tumor_fraction"]
    alpha = grouped.sum()
    n_spots = grouped.count()

    tile_purity = (alpha + 0.5 * cfg.k_prior) / (n_spots + cfg.k_prior)     # prior mean = 0.5 (agnostic)
    tile_purity = tile_purity.rename("tile_purity")

    final_df = df.merge(tile_purity, on="tile_id", how="left")

    metadata = {"mean_tile_purity": float(tile_purity.mean()),
                "median_tile_purity": float(tile_purity.median()),
                "tile_purity_std": float(tile_purity.std())
                }
    
    msg = f"Bayesian tile purtiy computed (k={cfg.k_prior}): mean={metadata['mean_tile_purity']:.3f}"
    logger.info(msg)

    # warning is tumor purity is very low
    if metadata['mean_tile_purity'] < cfg.mean_purity_threshold:
        msg = f"Mean tile purity is very low: {metadata['mean_tile_purity']}"
        logger.warning(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tile_purity", "FLAG", msg, metadata)
    else:
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tile_purity", "OK", 
                                  f"Mean tile purity is above {cfg.mean_purity_threshold * 100}%",
                                  metadata)    

    return final_df, metadata


def filter_tumor_tiles(df,        
                       patient_id,
                       qc_tracker=None,
                       cfg=PREPROCESSING_CONFIG
                       ):
    """
    Filter tumor tiles (purity + min_spots)
    """

    tiles_stats = df.groupby(["tile_row", "tile_col"]).agg(
        tile_id=("tile_id", "first"),
        tile_purity=("tile_purity", "first"),
        n_spots=("barcode", "count")
    ).reset_index()

    tumor_threshold = cfg.tumor_purity_threshold
    min_spots = cfg.min_spots_per_tile

    tumor_tiles = tiles_stats.query(
        "tile_purity >= @tumor_threshold and n_spots >= @min_spots"
    )

    metadata = {
        "tiles_before_filter": len(tiles_stats),
        "n_tumor_tiles": len(tumor_tiles),
        "has_tumor_tiles": len(tumor_tiles) > 0
    }

    logger.info(f"Filtered from {metadata['tiles_before_filter']} to {metadata['n_tumor_tiles']} tumor tiles")

    # add QC records
    if metadata['n_tumor_tiles'] == 0:
        msg = f"No tumor tiles found! Check threshold={tumor_threshold}, min_spots={min_spots}"
        logger.error(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "EXCLUDE", msg, metadata)
        
    elif metadata['n_tumor_tiles'] < cfg.min_final_tumor_tiles:
        msg = f"Only {len(tumor_tiles)} tumor tiles - insufficient for MIL training"
        logger.warning(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "FLAG", msg, metadata)

    else:
            qc_tracker.add_record(patient_id, "filter_tumor_tiles", "OK", 
                                 f"Found {metadata['n_tumor_tiles']} tumor tiles", metadata)

    return tumor_tiles, metadata
