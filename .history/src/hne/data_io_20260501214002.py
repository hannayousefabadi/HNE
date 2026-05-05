
import scanpy as sc
from PIL import Image
import json

from hne.paths import VISIUM_ST, VISIUM_INFO

def load_visium():
    adata = sc.read_h5ad(VISIUM_ST / )

    return adata


def load_spots():
    spots = pd.read_csv(VISIUM_INFO / "tissue_positions.csv")

    return spots


def load_scale_factor():    
    with open(VISIUM_INFO / "scalefactors_json.json") as f:
        scalefactors = json.load(f)

    return scalefactors    


def load_he_image():    
    hne_img = Image.open(VISIUM_INFO / "tissue_hires_image.png") 

    return hne_img



# 1. Load ST data (AnnData + Visium coordinates)


# 11. Save tile‑level signature matrix
save_tile_features()
