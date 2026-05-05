# 2. Compute/attach tumor fraction per spot
import pandas as pd

def attach_tumor_fraction(spots, vis, logger=None, patient_id):
    """
    Compute tumor fraction per spot
    Returns:
        merged_df
        metadata_dict
    """
    if logger is None:
        # fall back to a dummy logger
        import logging
        logger = logging.getLogger(__name__)
    
    
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

    logger.info(f"Patient {patient_id} has {metadata['n_spots_in_tissue']} tissue spots,"
                f"{metadata['n_spots_missin_tumor_fraction']} spots with missing tumor fraction")
    
    if metadata['n_spots_missin_tumor_fraction'] > metadata['n_spots_in_tissue'] * 10:
        logger.warning(f"High proportion of spots missing tumor fraction: "
                       f"{metadata['n_spots_missin_tumor_fraction']}/{metadata['n_spots_in_tissue']}")

    return merged, metadata



def add_tile_coordinates(scales, 
                         tile_size,     # ≈ 1 mm in hires image
                         merged,
                         logger=None
                         ):
    """
    Derive tile coordinates from spot positions
    Returns:
        df
        metadata
    """
    if logger is None:
        # fall back to a dummy logger
        import logging
        logger = logging.getLogger(__name__)

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
        "fullres_pixel_size": round(fullres_pixel_size, 3),
        "hires_pixel_size": round(hires_pixel_size, 3)
    }
    
    logger.info(f"Created {metadata['n_initial_tiles']} initial tiles")

    # warning if very few tiles
    if metadata['n_initial_tiles'] < 100:
        logger.warning(f"Very few initial tiles created: {metadata['n_initial_tiles']}")    


    return merged, metadata



def compute_tile_purity(
        df: pd.DataFrame,
        k=2,
        logger=None
        ) -> pd.DataFrame:
    """
    Computes Bayesian tumor purity per tile. 
    Returns:
        final_df
        metadata
    """
    if logger is None:
        # fallback to a dummy logger
        import logging
        logger = logging.getLogger(__name__)

    grouped = df.groupby("tile_id")["tumor_fraction"]
    alpha = grouped.sum()
    n_spots = grouped.count()

    tile_purity = (alpha + 0.5 * k) / (n_spots + k)     # prior mean = 0.5 (agnostic)
    tile_purity = tile_purity.rename("tile_purity")

    final_df = df.merge(tile_purity, on="tile_id", how="left")

    metadata = {"k_value": k,
                "mean_tile_purtiy": float(tile_purity.mean()),
                "median_tile_purity": float(tile_purity.median()),
                "tile_purity_std": float(tile_purity.std())
                }
    
    logger.info(f"Bayesian tile purtiy computed (k={k}): mean={metadata['mean_tile_purtiy']:.3f}")

    # warning is tumor purity is very low
    if metadata['mean_tile_purtiy'] < 0.3:
        logger.warning(f"Mean tile purity is very low: {metadata['mean_tile_purt']}")

    return final_df, metadata


def filter_tumor_tiles(df, 
                       tumor_threshold=0.3, # tile at least has 30% tumor purity
                       min_spots=40,        # tile at least has 40 spots
                       patient_id,
                       logger=None,
                       qc_tracker=None
                       ):
    """
    Filter tumor tiles (purity + min_spots)
    Returns:
        tumor_tiles
        metadata
    """
    if logger is None:
        # fallback to a dummy logger
        import logging
        logger = logging.getLogger(__name__)

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

    # ERROR & WARNING: critical check for MIL
    if metadata['n_tumor_tiles'] == 0:
        msg = f"No tumor tiles found! Check threshold={tumor_threshold}, min_spots={min_spots}"
        logger.error(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tumor_filter", "ERROR", msg)
        return tumor_tiles, metadata
    elif metadata['n_tumor_tiles'] < 10:
        msg = f"Only {len(tumor_tiles)} tumor tiles - insufficient for MIL training"
        logger.warning(msg)
        if qc_tracker:
            qc_tracker.add_record(patient_id, "tumor_filter", "WARNING", msg)


    return tumor_tiles, metadata






