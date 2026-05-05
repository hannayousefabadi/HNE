
import scanpy as sc
import pandas as pd
from PIL import Image
import json

from hne.paths import PatienPaths

def load_visium(paths: PatienPaths):
    h5ad_file = f"{paths.patient_id}a_vis_c2l_annots.h5ad"
    return sc.read_h5ad(paths.visium_st / h5ad_file)


def load_spots(PatienPaths):
    return pd.read_csv(PatienPaths.visium_info / "tissue_positions.csv")


def load_scale_factor(PatienPaths):    
    with open(PatienPaths.visium_info / "scalefactors_json.json") as f:
        return json.load(f)
    
def load_he_image(PatienPaths):     
    return Image.open(PatienPaths.visium_info / "tissue_hires_image.png")



# 11. Save tile‑level signature matrix
def save_tile_features():
