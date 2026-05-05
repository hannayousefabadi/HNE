
import scanpy as sc
from PIL import Image
import json
from pathlib import Path

from hne.paths import VISIUM_ST, VISIUM_INFO

def load_visium(VISIUM_ST, paths):
    return sc.read_h5ad(VISIUM_ST / f"{paths.base.name}a_vis_c2l_annots.h5ad")


def load_spots(VISIUM_INFO):
    return pd.read_csv(VISIUM_INFO / "tissue_positions.csv")


def load_scale_factor(VISIUM_INFO):    
    with open(VISIUM_INFO / "scalefactors_json.json") as f:
        return json.load(f)
    

def load_he_image(VISIUM_INFO):     
    return Image.open(VISIUM_INFO / "tissue_hires_image.png")


# 1. Load ST data (AnnData + Visium coordinates)


# 11. Save tile‑level signature matrix
save_tile_features()
