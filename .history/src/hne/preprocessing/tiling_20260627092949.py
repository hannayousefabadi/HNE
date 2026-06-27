from hne.utils import get_logger
from hne.core.paths import TILES
from pathlib import Path

logger = get_logger()

def crop_and_save_tiles(tumor_tiles, tile_size, hne_img, patient_id):
    """
    Generate & save image tiles (H&E crops)
    """
    tiles = []
    tiles_path = Path(TILES) / patient_id
    tiles_path.mkdir(parents=True, exist_ok=True)

    for _, row in tumor_tiles.iterrows():

        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])

        left = tile_col * tile_size
        upper = tile_row * tile_size
        right = (tile_col + 1) * tile_size
        lower = (tile_row + 1) * tile_size

        tile_img = hne_img.crop((left, upper, right, lower))
        tiles.append(tile_img)

        tile_id = f"tile_r{tile_row}_c{tile_col}"
        tile_img.save(tiles_path / f"{tile_id}.png", dpi=(600, 600))

        # metadata
        metadata = {
            "n_tumor_tiles_saved": len(tiles),
            "tiles_path": str(tiles_path)
        }

    logger.info(f"Saved {len(tiles)} tumor tiles for patient {patient_id}")

    return tiles, metadata
    

