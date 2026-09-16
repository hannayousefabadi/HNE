"""src/hne/preprocessing/aggregation.py"""

import pandas as pd
import numpy as np

from hne.utils import get_logger
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG

logger = get_logger()

def aggregate_signatures(spots_df, sig_cols, tile_size, tumor_tiles_df, cfg=PREPROCESSING_CONFIG):
    """
    Aggregate spot-level ssGSEA signatures to tile-level by taking the mean.
    Tile scores maintain their raw enrichment scale across all patients.
    """
    if len(spots_df) == 0:
        logger.warning("No spots data provided for aggregation")

    if len(tumor_tiles_df) == 0:
        logger.warning("No tumor tiles provided - returning empty result")
        
    # aggregate tiles metadata    
    tiles_mdata = spots_df.groupby("tile_id").agg(
        tile_row=("tile_row", "first"),
        tile_col=("tile_col", "first"),
        tile_purity=("tile_purity", "first"),
        n_spots=("barcode", "count")
    )

    # aggregate ssGSEA scores across spots in each tile
    tiles_sig = spots_df.groupby("tile_id")[sig_cols].mean()
    tiles_sig = tiles_mdata.merge(tiles_sig, on="tile_id", how="left")

    # add pixel bounding boxes
    # x = horizontal (cols), y = vertical (rows)
    tiles_sig["x_min_fullres"] = tiles_sig["tile_col"] * tile_size
    tiles_sig["y_min_fullres"] = tiles_sig["tile_row"] * tile_size
    tiles_sig["x_max_fullres"] = tiles_sig["x_min_fullres"] + tile_size
    tiles_sig["y_max_fullres"] = tiles_sig["y_min_fullres"] + tile_size

    # filter to tumor tiles
    tumor_tiles_id = set(tumor_tiles_df["tile_id"])
    tiles_sig_tumor = tiles_sig.loc[tiles_sig.index.isin(tumor_tiles_id)].copy()
    tiles_sig_tumor = tiles_sig_tumor.reset_index()    # turning 'tile_id' into a column

    # metadata
    metadata = {
        "n_total_tiles_aggregated": len(tiles_sig),
        "n_tumor_tiles_aggregated": len(tiles_sig_tumor),
        "avg_spots_per_tile": round(float(tiles_sig_tumor['n_spots'].mean()), 2) if len(tiles_sig_tumor) > 0 else 0.0
    }

    logger.info(f"Aggregated {metadata['n_tumor_tiles_aggregated']} tumor tiles "
            f"with avg {metadata['avg_spots_per_tile']} spots per tile")

    if metadata['n_total_tiles_aggregated'] > 0:
        pct_tumor = (metadata["n_tumor_tiles_aggregated"] / metadata["n_total_tiles_aggregated"]) * 100
        logger.debug(f"{pct_tumor:.2f}% of tiles are tumor tiles")

    if metadata["n_tumor_tiles_aggregated"] < cfg.min_final_tumor_tiles:
        logger.warning(f"Very few tumor tiles after aggregation: "
                       f"{metadata['n_tumor_tiles_aggregated']} tiles")  

    return tiles_sig_tumor, metadata


def binary_scores(sig_cols, tiles_sig_tumor, cfg=PREPROCESSING_CONFIG):
    """
    Generate binary signature calls directly from raw ssGSEA scores.
    Tiles with enrichment >= quantile_threshold (default top 25%) are flagged as 1.0, else 0.0.
    Retains NaN for missing signatures to allow loss masking downstream.
    """
    if len(tiles_sig_tumor) == 0:
        return tiles_sig_tumor

    tiles_sig_tumor = tiles_sig_tumor.copy()
    binary_data = {}

    for col in sig_cols:
        series = tiles_sig_tumor[col]

        # handle columns that are completely missing (all NaN)
        if series.isna().all():
            binary_data[f"{col}_binary"] = np.nan
            logger.debug(f"{col} is entirely NaN - preserving NaN across all binary calls")
            continue

        cutoff = series.quantile(cfg.binary_quantile_threshold)
        
        # populate dict: 1.0 if >= cutoff, 0.0 if < cutoff, NaN if missing
        binary_data[f"{col}_binary"] = np.where(
            series.isna(),
            np.nan,
            (series >= cutoff).astype(float)
        )

        logger.debug(f"{col} binary cutoff (p{int(cfg.binary_quantile_threshold * 100)}): {cutoff:.4f}")

    # concat all binary columns at once to avoid DataFrame fragmentation
    binary_df = pd.DataFrame(binary_data, index=tiles_sig_tumor.index)
    tiles_sig_tumor = pd.concat([tiles_sig_tumor, binary_df], axis=1)

    return tiles_sig_tumor