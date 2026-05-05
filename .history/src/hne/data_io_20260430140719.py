
import scanpy as sc
from PIL import Image
import json


def load_visium():
    


adata = sc.read_h5ad(VISIUM_ST / "CH_L_282a_vis_c2l_annots.h5ad")

hne_img = Image.open(VISIUM_INFO / "tissue_hires_image.png")
spots = pd.read_csv(VISIUM_INFO / "tissue_positions.csv")

with open(VISIUM_INFO / "scalefactors_json.json") as f:
    scalefactors = json.load(f)







# 1. Load ST data (AnnData + Visium coordinates)

load_visium()
load_spots()
load_scalefactors()
load_he_image()

# 11. Save tile‑level signature matrix
save_tile_features()
