
from hne.paths import TILES, PATIENT_IDS

def crop_and_save_tiles(tumor_tiles, TILE_SIZE):
    """
    Generate & save image tiles (H&E crops)
    """

    tiles = []

    for _, row in tumor_tiles.iterrows():

        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])

        left = tile_col * TILE_SIZE
        upper = tile_row * TILE_SIZE
        right = (tile_col + 1) * TILE_SIZE
        lower = (tile_row + 1) * TILE_SIZE

        tile_img = hne_img.crop((left, upper, right, lower))
        tiles.append(tile_img)

        tile_id = f"tile_r{tile_row}_c{tile_col}"
        tile_img.save(f"TILES / PATIENT_ID /{tile_id}.png")




