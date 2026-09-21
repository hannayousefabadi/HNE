"""tests/debug_feat_extraction.py"""
import numpy as np
import pandas as pd
from PIL import Image
from hne.core.paths import PATIENTS, TILES_SIGNATURE_MATRIX, PREPROCESSING_QC_REPORTS
from hne.core.data_io import load_he_slide

pid = "CH_L_275a"
tiles = pd.read_csv(TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{pid}.csv")
paths = PATIENTS[pid]

print(f"Testing slide loading for {pid}...")
with load_he_slide(paths) as slide:
    print(f"Slide Type: {type(slide)}")
    print(f"Slide Dimensions (W, H): {slide.dimensions}")
    
    row0 = tiles.iloc[0]
    row1 = tiles.iloc[1]
    x0, y0 = int(row0["x_min_fullres"]), int(row0["y_min_fullres"])
    x1, y1 = int(row1["x_min_fullres"]), int(row1["y_min_fullres"])
    
    print(f"Tile 0 Coords: ({x0}, {y0})")
    print(f"Tile 1 Coords: ({x1}, {y1})")
    
    # Read two 224x224 crops
    crop0 = slide.read_region((x0, y0), 0, (224, 224)).convert("RGB")
    crop1 = slide.read_region((x1, y1), 0, (224, 224)).convert("RGB")
    
    # Save crops to disk to inspect them visually
    crop0.save("debug_tile0.png")
    crop1.save("debug_tile1.png")
    print("Saved debug_tile0.png and debug_tile1.png")
    
    arr0 = np.array(crop0)
    arr1 = np.array(crop1)
    
    print(f"Crop 0 RGB Mean: {arr0.mean():.2f}, Std: {arr0.std():.2f}")
    print(f"Crop 1 RGB Mean: {arr1.mean():.2f}, Std: {arr1.std():.2f}")
    print(f"Are raw crops identical? {np.allclose(arr0, arr1)}")