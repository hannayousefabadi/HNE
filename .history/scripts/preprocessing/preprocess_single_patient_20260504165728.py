from hne.paths import PATIENTS
from hne.data_io import load_visium, load_spots, load_scale_factor, load_he_image
from hne.tumor_purity import attach_tumor_fraction, add_tile_coordinates, compute_tile_purity, filter_tumor_tiles

paths = PATIENTS["CH_L_282"]

TILE_SIZE = 100   # ≈ 1 mm in hires image


# 1. Load ST data
adata = load_visium(paths)
spots = load_spots(paths)
scales = load_scale_factor(paths)
img = load_he_image(paths)


# 2. Compute/attach tumor fraction per spot
# 3. Merge spots metadata and tumor fractions
merged = attach_tumor_fraction(spots, adata)

# 4. Derive tile coordinates from spot positions
df = add_tile_coordinates(scale=scales, tile_size=TILE_SIZE, merged=merged)

# 5. Compute Bayesian tile purity from spot tumor fractions
final_df = compute_tile_purity(df, k=2)

# 6. Filter tumor tiles (purity + min_spots)
tumor_tiles = filter_tumor_tiles(final_df,
                   tumor_threshold=0.3, # tile at least has 30% tumor purity
                   min_spots=40         # tile at least has 40 spots  
                )

# 7. Generate & save image tiles (H&E crops)

# 8. Compute pathway signatures in spot space
# 9. Aggregate spot signatures to tile-level
# 10. Apply z‑score & binary calls
# 11. Save tile‑level signature matrix (MIL‑ready)
# 12. QC metrics (variation, sparsity, consistency, redundancy)




