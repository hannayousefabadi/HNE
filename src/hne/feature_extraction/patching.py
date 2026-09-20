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
    valid_tissue = (gray >= 15) & (gray <= 220)
    return float(valid_tissue.sum() / gray.size) >= min_tissue_fraction


def stream_patches_for_tile(
    slide: openslide.OpenSlide,
    x0: int,
    y0: int,
    tile_size_px_fullres: int,  # tile size in px
    fullres_pixel_size: float,
    spec: PatchSpec,
    min_tissue_fraction: float = 0.5,
) -> Generator[Image.Image, None, None]:
    """
    Streams model-ready patches one by one without accumulating PIL objects in memory.
    No overlap between patches by default design. but the full coverage of each tile is 
    guaranteed (the remainder strip is covered by one extra patch per axis, which overlaps its neighbor)
    """
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
            # Ensure coordinates fit strictly inside the WSI canvas
            if px < 0 or py < 0 or (px + patch_px_native) > slide_w or (py + patch_px_native) > slide_h:
                continue

            # using the read_region to stream just the patch, without ever materializing the whole decoded image in memory,
            crop = slide.read_region((px, py), 0, (patch_px_native, patch_px_native)).convert("RGB")

            if not _has_enough_tissue(crop, min_tissue_fraction):   # True
                crop.close()
                continue
                
            # resize patches to models' expected pixel size    
            resized = crop.resize((spec.patch_size_px, spec.patch_size_px), Image.BICUBIC)
            crop.close()
            yield resized

