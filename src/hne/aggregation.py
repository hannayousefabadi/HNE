
def aggregate_signatures(spots_df, sig_cols, tile_size, tumor_tiles):
    """
    Aggregate spot signatures to tile-level
    """
    tiles_mdata = spots_df.groupby("tile_id").agg(
    tile_row=("tile_row", "first"),
    tile_col=("tile_col", "first"),
    tile_purity=("tile_purity", "first"),
    n_spots=("barcode", "count")
    )

    tiles_sig = spots_df.groupby("tile_id")[sig_cols].mean()
    tiles_sig = tiles_mdata.merge(tiles_sig, on="tile_id", how="left")

    # x = horizontal (cols), y = vertical (rows)
    tiles_sig['x_min_hires'] = tiles_sig['tile_col'] * tile_size
    tiles_sig['y_min_hires'] = tiles_sig['tile_row'] * tile_size
    tiles_sig['x_max_hires'] = tiles_sig['x_min_hires'] + tile_size
    tiles_sig['y_max_hires'] = tiles_sig['y_min_hires'] + tile_size

    tumor_tiles_id = set(tumor_tiles["tile_id"])

    tiles_sig_tumor = tiles_sig.loc[tiles_sig.index.isin(tumor_tiles_id)].copy()
    print(len(tiles_sig_tumor), "tumor tiles with signatures")

    return tiles_sig_tumor



def zscore_and_binary(sig_cols, tiles_sig_tumor):
    """
    Apply z-score and binary calls
    """
    BINARY_THRESHOLD = 1.0

    for col in sig_cols:
        # computing z-score
        z = (tiles_sig_tumor[col] - tiles_sig_tumor[col].mean()) / tiles_sig_tumor[col].std()
        tiles_sig_tumor[f"{col}_z"] = z

        tiles_sig_tumor[f"{col}_binary"] = (z >= BINARY_THRESHOLD).astype(int) 

    return tiles_sig_tumor    
