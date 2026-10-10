from pathlib import Path
from PIL import Image
import openslide

from hne.utils import get_logger
from hne.core.paths import TILES


logger = get_logger()

def drop_tiles_outside_slide(tumor_tiles, tile_size: int, slide_width: int, slide_height: int, patient_id: str):
    """
    Keep only the tiles that lie fully inside the H&E scan. The Visium capture area can reach
    past the edge of the scan, so a few spots, and the tiles built on them, have no image under
    them. If most tiles are outside, the wrong image was opened and this raises.

    Returns the kept tiles and the metadata.
    """
    inside = (((tumor_tiles["tile_col"] + 1) * tile_size <= slide_width) &
              ((tumor_tiles["tile_row"] + 1) * tile_size <= slide_height))
    n_outside = int((~inside).sum())
    if n_outside > 0.5 * len(tumor_tiles):
        raise ValueError(
            f"{patient_id}: {n_outside} of {len(tumor_tiles)} tumor tiles are not inside the "
            f"{slide_width}x{slide_height} slide. The opened image is not the one the fullres coordinates refer to."
        )
    if n_outside:
        logger.warning(f"{patient_id}: dropped {n_outside} tumor tile(s) that reach past the edge of the H&E scan")
    kept = tumor_tiles[inside].copy()
    return kept, {"n_tiles_outside_slide": n_outside, "n_tumor_tiles": len(kept), "has_tumor_tiles": len(kept) > 0}


def crop_and_save_tiles(tumor_tiles, 
                        tile_size: int, 
                        slide: openslide.OpenSlide, 
                        patient_id: str):
    """
    Generate & save image tiles (H&E crops). stream tile crops directly from OpenSlide
    without loading the full slide into the RAM. Each tile is written to disk and released
    before the next one is read, so memory use does not grow with the number of tiles.

    Returns the number of tiles saved and the metadata.
    """
    n_saved = 0
    tiles_path = Path(TILES) / patient_id
    tiles_path.mkdir(parents=True, exist_ok=True)

    slide_w, slide_h = slide.dimensions

    for _, row in tumor_tiles.iterrows():
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])

        left = tile_col * tile_size
        upper = tile_row * tile_size

        # a tile on the slide border may extend past it: crop what exists and pad the rest
        actual_w = min(tile_size, max(0, slide_w - left))
        actual_h = min(tile_size, max(0, slide_h - upper))

        # a tile that starts outside the slide means this is not the full-resolution H&E
        if actual_w == 0 or actual_h == 0:
            raise ValueError(
                f"{patient_id}: tile {tile_row}-{tile_col} starts at ({left}, {upper}) px, outside the "
                f"{slide_w}x{slide_h} slide. The opened image is not the one the fullres coordinates refer to."
            )

        # OpenSlide.read_region reads (left, top) at level 0 (full resolution)
        tile_img = slide.read_region((left, upper), 0, (actual_w, actual_h)).convert("RGB")

        # if edge tile is smaller than full tile_size, pad to uniform square
        if actual_w != tile_size or actual_h != tile_size:
            padded_img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            padded_img.paste(tile_img, (0, 0))
            tile_img = padded_img

        tile_id = f"tile_r{tile_row}_c{tile_col}"
        tile_img.save(tiles_path / f"{tile_id}.png")
        tile_img.close()
        n_saved += 1

    metadata = {"tiles_path": str(tiles_path), "n_tile_images_saved": n_saved}
    logger.info(f"Saved {n_saved} tumor tiles for patient {patient_id}")
    return n_saved, metadata

