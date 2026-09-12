from pathlib import Path
from PIL import Image
import openslide

from hne.utils import get_logger
from hne.core.paths import TILES


logger = get_logger()

def crop_and_save_tiles(tumor_tiles, 
                        tile_size: int, 
                        slide: openslide.OpenSlide, 
                        patient_id: str):
    """
    Generate & save image tiles (H&E crops). stream tile crops directly from OpenSlide
    without loading the full slide into the RAM.
    """
    tiles = []
    tiles_path = Path(TILES) / patient_id
    tiles_path.mkdir(parents=True, exist_ok=True)

    slide_w, slide_h = slide.dimensions

    for _, row in tumor_tiles.iterrows():
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])

        left = tile_col * tile_size
        upper = tile_row * tile_size

        # clamp boundaries to prevent out-of-bounds artifact creation
        actual_w = min(tile_size, max(0, slide_w - left))
        actual_h = min(tile_size, max(0, slide_h - upper))

        if actual_w == 0 or actual_h == 0:
            continue

        # OpenSlide.read_region reads (left, top) at level 0 (full resolution)
        tile_img = slide.read_region((left, upper), 0, (actual_w, actual_h)).convert("RGB")

        # if edge tile is smaller than full tile_size, pad to uniform square
        if actual_w != tile_size or actual_h != tile_size:
            padded_img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            padded_img.paste(tile_img, (0, 0))
            tile_img = padded_img

        tiles.append(tile_img)

        tile_id = f"tile_r{tile_row}_c{tile_col}"
        tile_img.save(tiles_path / f"{tile_id}.png")

    metadata = {"tiles_path": str(tiles_path)}
    logger.info(f"Saved {len(tiles)} tumor tiles for patient {patient_id}")
    return tiles, metadata

