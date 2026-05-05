
import scanpy as sc
import pandas as pd
from PIL import Image
import json

from hne.paths import PatienPaths, TILE_FEATURES, PATIENT_IDS

def load_visium(paths: PatienPaths):
    h5ad_file = f"{paths.patient_id}a_vis_c2l_annots.h5ad"
    return sc.read_h5ad(paths.visium_st / h5ad_file)


def load_spots(paths: PatienPaths):
    return pd.read_csv(paths.visium_info / "tissue_positions.csv")


def load_scale_factor(paths: PatienPaths):    
    with open(paths.visium_info / "scalefactors_json.json") as f:
        return json.load(f)
    
def load_he_image(paths: PatienPaths):     
    return Image.open(PatienPaths.visium_info / "tissue_hires_image.png")


def save_tile_features(tiles_sig_tumor):
    """
    Save tile‑level signature matrix
    """
    tiles_sig_tumor.to_csv(TILE_FEATURES / PATIENT_IDS / "tiles_signature_matrix_CH_L_282a.csv", index=False)