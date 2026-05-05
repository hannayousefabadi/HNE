
import scanpy as sc
from PIL import Image
import json
from pathlib import Path

from hne.paths import PatienPaths

def load_visium(PatienPaths):
    return sc.read_h5ad(PatienPaths.visium_st / f"{PatienPaths.base.name}a_vis_c2l_annots.h5ad")


def load_spots(PatienPaths):
    return pd.read_csv(PatienPaths.visium_info / "tissue_positions.csv")


def load_scale_factor(PatienPaths):    
    with open(PatienPaths.visium_info / "scalefactors_json.json") as f:
        return json.load(f)
    

def load_he_image(PatienPaths):     
    return Image.open(PatienPaths.visium_info / "tissue_hires_image.png")


# 1. Load ST data (AnnData + Visium coordinates)


# 11. Save tile‑level signature matrix
save_tile_features()
