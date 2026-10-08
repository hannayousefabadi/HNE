"""src/hne/feature_extraction/patching.py"""

from dataclasses import dataclass
from typing import Generator, Tuple
from PIL import Image
import openslide
import numpy as np


@dataclass
class PatchSpec:
    """FMs-specific patch requirements."""
    patch_size_px: int    # model pixel input size, e.g. 224 px
    patch_fov_um: float   # physical tissue area (field of view), e.g. 112 µm
    stride_um: float = None  # defaults to non-overlapping (== patch_fov_um)

    def __post_init__(self):
        if self.stride_um is None:
            self.stride_um = self.patch_fov_um


PATCH_SPECS = {
    "phikon_v2": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "virchow2": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "uni2_h": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "gigapath": PatchSpec(patch_size_px=256, patch_fov_um=128.0),
    "conch_v15": PatchSpec(patch_size_px=512, patch_fov_um=256.0),
}


def _tile_positions(start: int, tile_size_px: int, patch_px: int, stride_px: int) -> list[int]:
    """
    Ensure the tile's full coverage considering different image and patch sizes
    the final position is snapped back to the tile edge instead
    of being dropped, at the cost of a small overlap with the previous patch.
    Causing small amount of double-coverage at the trailing edges only.
    """
    if patch_px > tile_size_px:
        return [start]
    # generate the normal grids
    positions = list(range(start, start + tile_size_px - patch_px + 1, stride_px))
    # `last_valid_start` is that last position where a patch still fits inside the tile
    last_valid_start = start + tile_size_px - patch_px
    # if the regular stride grid didn't exactly land on `last_valid_start`, it appends it
    if not positions or positions[-1] != last_valid_start:
        positions.append(last_valid_start)
    return positions


def _has_enough_tissue(crop: Image.Image, min_tissue_fraction: float = 0.5) -> bool:
    """
    Robust tissue/background filter:
    Filters out both white/glass background (>220) AND out-of-bounds/black areas (<15).
    """
    gray = np.array(crop.convert("L"))  # converting RGB to grayscale
    valid_tissue = (gray <= 220).sum()
    return float(valid_tissue / gray.size) >= min_tissue_fraction


def require_fullres_slide(patient_id: str, slide_width: int, slide_height: int, tiles_df) -> None:
    """
    Tile coordinates (`x_min_fullres` ...) are pixels of the full-resolution H&E that Space
    Ranger registered the spots to. The slide opened for extraction must be that image.
    Raises if any tile starts outside it, which is what happens when a CytAssist image or
    any downscaled image is opened instead. Coordinates are never rescaled to make them fit.
    A tile on the slide border may extend past it; its outside patches are skipped and counted.
    """
    far_x = float(tiles_df["x_min_fullres"].max())
    far_y = float(tiles_df["y_min_fullres"].max())
    if far_x >= slide_width or far_y >= slide_height:
        raise ValueError(
            f"{patient_id}: the opened slide is {slide_width}x{slide_height} px but tiles start as far as "
            f"({far_x:.0f}, {far_y:.0f}) px. This is not the full-resolution H&E the coordinates refer to "
            f"(a 3000x3000 slide is the CytAssist image). Check the cohort manifest."
        )


def new_patch_stats() -> dict:
    """Counters filled in by stream_patches_for_tile for one tile."""
    return {"n_positions": 0, "n_out_of_bounds": 0, "n_read_error": 0, "n_low_tissue": 0, "n_patches_used": 0}


def stream_patches_for_tile(
    slide: openslide.OpenSlide,
    x0: int,
    y0: int,
    tile_size_px_fullres: int,  # tile size in px
    fullres_pixel_size: float,
    spec: PatchSpec,
    min_tissue_fraction: float = 0.5,
    stats: dict = None,
) -> Generator[Image.Image, None, None]:
    """
    Streams model-ready patches one by one without accumulating PIL objects in memory.
    No overlap between patches by default design. but the full coverage of each tile is 
    guaranteed (the remainder strip is covered by one extra patch per axis, which overlaps its neighbor)

    A patch whose window is not fully inside the slide is skipped and counted, never moved
    to the slide edge. Pass `stats` (from new_patch_stats) to get the per-tile counts.
    """
    if stats is None:
        stats = new_patch_stats()
    slide_w, slide_h = slide.dimensions
    # converting a physical measurement (µm) into native pixels  for this specific patient's
    # fullres image using the patient's o‍wn mpp (fullres_pixel_size)
    patch_px_native = max(1, round(spec.patch_fov_um / fullres_pixel_size))     # unit: pixels, how many pixels
    # how wide/tall do I need to crop from this patient's tiff to capture 112 µm of real tissue
    stride_px_native = max(1, round(spec.stride_um / fullres_pixel_size))   # unit: pixels, how far to move between crops (overlap)

    y_positions = _tile_positions(y0, tile_size_px_fullres, patch_px_native, stride_px_native)
    x_positions = _tile_positions(x0, tile_size_px_fullres, patch_px_native, stride_px_native)

    for py in y_positions:
        for px in x_positions:
            stats["n_positions"] += 1

            # 1. a window outside the slide means the coordinates are wrong: skip and count
            if px < 0 or py < 0 or px + patch_px_native > slide_w or py + patch_px_native > slide_h:
                stats["n_out_of_bounds"] += 1
                continue

            # 2. read the native crop
            try:
                crop = slide.read_region((px, py), 0, (patch_px_native, patch_px_native)).convert("RGB")
            except Exception:
                stats["n_read_error"] += 1
                continue

            # 3. check tissue content
            if not _has_enough_tissue(crop, min_tissue_fraction):
                stats["n_low_tissue"] += 1
                crop.close()
                continue
                
            # resize patches to models' expected pixel size    
            resized = crop.resize((spec.patch_size_px, spec.patch_size_px), Image.BICUBIC)
            crop.close()
            stats["n_patches_used"] += 1
            yield resized
