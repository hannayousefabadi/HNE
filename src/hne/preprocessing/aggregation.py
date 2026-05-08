
def aggregate_signatures(spots_df, sig_cols, tile_size, tumor_tiles_df, logger=None):
    """
    Aggregate spot signatures to tile-level
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # aggregate tiles metadata    
    tiles_mdata = spots_df.groupby("tile_id").agg(
        tile_row=("tile_row", "first"),
        tile_col=("tile_col", "first"),
        tile_purity=("tile_purity", "first"),
        n_spots=("barcode", "count")
    )

    # aggregate signatures (mean across spots in tiles)
    tiles_sig = spots_df.groupby("tile_id")[sig_cols].mean()
    tiles_sig = tiles_mdata.merge(tiles_sig, on="tile_id", how="left")

    # add pixel coordinates
    # x = horizontal (cols), y = vertical (rows)
    tiles_sig['x_min_hires'] = tiles_sig['tile_col'] * tile_size
    tiles_sig['y_min_hires'] = tiles_sig['tile_row'] * tile_size
    tiles_sig['x_max_hires'] = tiles_sig['x_min_hires'] + tile_size
    tiles_sig['y_max_hires'] = tiles_sig['y_min_hires'] + tile_size

    # filter to tumor tiles
    tumor_tiles_id = set(tumor_tiles_df["tile_id"])
    tiles_sig_tumor = tiles_sig.loc[tiles_sig.index.isin(tumor_tiles_id)].copy()

    # metadata
    metadata = {
        "n_total_tiles_aggregated": len(tiles_sig),
        "n_tumor_tiles_aggregated": len(tiles_sig_tumor),
        "avg_spots_per_tiles": round(float(tiles_sig_tumor['n_spots'].mean()), 2)
    }

    if metadata['n_total_tiles_aggregated'] > 0:
        pct_tumor = (metadata["n_tumor_tiles_aggregated"] / metadata["n_total_tiles_aggregated"]) * 100
        logger.debug(f"{pct_tumor:.2f}% of tiles are tumor tiles")

    return tiles_sig_tumor, metadata


def zscore_and_binary(sig_cols, tiles_sig_tumor):
    """
    Apply z-score normalization and binary calls to tile signatures
    
    """
    BINARY_THRESHOLD = 1.0

    for col in sig_cols:
        # computing z-score
        z = (tiles_sig_tumor[col] - tiles_sig_tumor[col].mean()) / tiles_sig_tumor[col].std()
        tiles_sig_tumor[f"{col}_z"] = z

        tiles_sig_tumor[f"{col}_binary"] = (z >= BINARY_THRESHOLD).astype(int) 

    return tiles_sig_tumor    
