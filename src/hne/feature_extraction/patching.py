"""src/hne/feature_extraction/patching.py"""

from dataclasses import dataclass
from PIL import Image
import openslide
import numpy as np


@dataclass
class PatchSpec:
    """FMs-specific patch requirements."""
    patch_size_px: int   # model pixel input size, e.g. 224 px
    patch_fov_um: float  # physical tissue area (filed of view), e.g. 112 µm
    stride_um: float = None # defaults to non-overlapping (== patch_fov_um)

    def __post_init__(self):
        if self.stride_um is None:
            self.stride_um = self.patch_fov_um

# registry - one entry per FM
PATCH_SPECS = {
    "phikon_v2": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "virchow2": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "uni2_h": PatchSpec(patch_size_px=224, patch_fov_um=112.0),
    "gigapath": PatchSpec(patch_size_px=256, patch_fov_um=128.0),
    "conch": PatchSpec(patch_size_px=512, patch_fov_um=256.0)
}

def extract_patches_for_tile(slide: openslide.OpenSlide,
                             x0: int, 
                             y0: int,
                             tile_size_px_fullres: int,     # tile size in px
                             fullres_pixel_size: float,
                             spec: PatchSpec,
                             min_tissue_fraction: float = 0.5
                             ) -> list[dict]:
    """Devide one tile (in fullres pixel coords) into model-ready patches."""
    # converting a physical measurement (µm) into native pixels  for this specific patient's
    # fullres image using the patient's own mpp (fullres_pixel_size)
    patch_px_native = round(spec.patch_fov_um / fullres_pixel_size)     # unit: pixels, how many pixels
    # wide/tall do I need to crop from this patient's tiff to capture 112 µm of real tissue
    stride_px_native = round(spec.stride_um / fullres_pixel_size)       # unit: pixels, how far to move
    # between crops (overlap)

    patches = []
    for py in range(y0, y0 + tile_size_px_fullres - patch_px_native + 1, stride_px_native):
        for px in range(x0, x0 + tile_size_px_fullres - patch_px_native + 1, stride_px_native):
            # cutting the right amount of real tissue
            crop = slide.read_region((px, py), 0, (patch_px_native, patch_px_native)).convert("RGB")

            if not _has_enough_tissue(crop, min_tissue_fraction):   # True
                continue
            
            # resize patches to models' expected pixel size
            crop = crop.resize((spec.patch_size_px, spec.patch_size_px), Image.BICUBIC)
            patches.append({"image": crop, "x": px, "y": py})

    return patches


def _has_enough_tissue(crop: Image.Image,
                       min_tissue_fraction: float
                       ) -> bool:
    """
    A simple tissue/background check
    """
    gray = np.array(crop.convert("L"))  # converting RGB patch to grayscale
    tissue_pixels = (gray < 220).sum()  # any pixel with grayscale intensity below 220 is assumed to be tissue
    return (tissue_pixels / gray.size) >= min_tissue_fraction
    




