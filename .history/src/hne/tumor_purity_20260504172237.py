# 2. Compute/attach tumor fraction per spot

import pandas as pd

def attach_tumor_fraction(spots, vis):
    """
    Compute tumor fraction per spot
    """
    in_tissue_spots = spots[spots["in_tissue"] == 1].copy()
    in_tissue_spots["barcode"] = in_tissue_spots["barcode"].astype(str)

    grp = vis.obsm["deconv_sample_level_custom1_frac"]
    tumor_fraction = grp["Tu_CH_L_282a_c01"] + grp["Tu_CH_L_282a_nos"]
    tumor_df = tumor_fraction.to_frame(name="tumor_fraction")
    tumor_df.index = tumor_df.index.astype(str)

    merged = in_tissue_spots.merge(tumor_df, how="left", left_on="barcode", right_index=True)

    # sanity check
    print(f"# of spots in tissue is {len(merged)}")
    print(f"# of spots without tumor fraction is {merged["tumor_fraction"].isna().sum()}")

    return merged



def add_tile_coordinates(scales, 
                         tile_size,     # ≈ 1 mm in hires image
                         merged
                         ):
    """
    Derive tile coordinates from spot positions
    """
    # spot_diameter = 55  # Visium spot diameter in µm

    # spot_diameter_fullres = scales["spot_diameter_fullres"]
    tissue_hires_scalef = scales["tissue_hires_scalef"]

    # fullres_pixel_size = spot_diameter / spot_diameter_fullres
    # hires_pixel_size = spot_diameter / (spot_diameter_fullres * tissue_hires_scalef)

    # sanity check
    # print(f"fullres pixel size is {fullres_pixel_size} µm / pixel")
    # print(f"hires pixel size is {hires_pixel_size} µm / pixel ≈ 9 µm/pixel ")

    merged["x_hires"] = merged["pxl_col_in_fullres"] * tissue_hires_scalef
    merged["y_hires"] = merged["pxl_row_in_fullres"] * tissue_hires_scalef
                                
    merged["tile_col"] = (merged["x_hires"] // tile_size).astype(int).copy()    # tile_col = index tiles (0,1,2,3,…) vertically
    merged["tile_row"] = (merged["y_hires"] // tile_size).astype(int).copy()    # tile_row = index tiles (0,1,2,3,…) horizontally
    merged["tile_id"] = merged["tile_row"].astype(str) + "-" + merged["tile_col"].astype(str)

    print(f"# of total tiles: {len(merged['tile_id'].unique())}")

    return merged



def compute_tile_purity(
        df: pd.DataFrame, 
        tile_id="tile_id", 
        tumor_fraction="tumor_fraction",
        k=2) -> pd.DataFrame:
    """
    Computes Bayesian tile purity
    posterior mean for tumor purity per tile
    """
    grouped = df.groupby(tile_id)[tumor_fraction]
    alpha = grouped.sum()
    n_spots = grouped.count()

    tile_purity = (alpha + 0.5 * k) / (n_spots + k)
    tile_purity = tile_purity.rename("tile_purity")

    final_df = df.merge(tile_purity, on=tile_id, how="left")

    return final_df



def filter_tumor_tiles(df, tumor_threshold=0.3, # tile at least has 30% tumor purity
                       min_spots=40         # tile at least has 40 spots
                       ):
    """
    Filter tumor tiles (purity + min_spots)
    """
        
    tiles_stats = df.groupby(["tile_row", "tile_col"]).agg(
        tile_id=("tile_id", "first"),
        tile_purity=("tile_purity", "first"),
        n_spots=("barcode", "count")
    ).reset_index()

    tiles_stats["n_spots"].describe()

    tumor_tiles = tiles_stats.query(
    "tile_purity >= @tumor_threshold and n_spots >= @min_spots"
    )

    return tumor_tiles





    