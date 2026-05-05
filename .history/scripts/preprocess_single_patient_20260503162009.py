from hne.paths import PATIENTS
from hne.data_io import load_visium, load_spots, load_scale_factor, load_he_image

paths = PATIENTS["CH_L_282"]

TILE_SIZE = 100 

adata = load_visium(paths)
spots = load_spots(paths)
scales = load_scale_factor(paths)
img = load_he_image(paths)



# 1. Tiling
tile_df = h.tiling.generate_tiles(paths['he_tif'])

# 2. Spot signatures
adata = h.spot_signatures.load_anndata(paths['h5ad'])
spots_df = h.spot_signatures.compute_signatures(adata)

# 3. QC
qc = h.signature_qc.run_full_qc(spots_df)

# 4. Aggregation
tiles_df = h.aggregation.aggregate_to_tiles(spots_df, tile_df)
tiles_df = h.aggregation.add_tile_coordinates(tiles_df)
tiles_df = h.aggregation.compute_binary_calls(tiles_df)

# 5. Save
h.aggregation.save_tile_features(tiles_df, paths['tile_feature_csv'])

print("Done processing", patient_id)
