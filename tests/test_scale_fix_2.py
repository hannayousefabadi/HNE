import numpy as np
import pandas as pd
from PIL import Image
from hne.core.paths import PATIENTS, TILES_SIGNATURE_MATRIX
from hne.core.data_io import load_he_slide

pid = "CH_L_275a"
scale = 0.0811897

tiles = pd.read_csv(TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{pid}.csv")
paths = PATIENTS[pid]

with load_he_slide(paths) as slide:
    print(f"Slide dimensions: {slide.dimensions}")
    row0 = tiles.iloc[0]
    row1 = tiles.iloc[1]
    
    # Scale coordinates to the 3000x3000 image canvas
    x0, y0 = int(row0["x_min_fullres"] * scale), int(row0["y_min_fullres"] * scale)
    x1, y1 = int(row1["x_min_fullres"] * scale), int(row1["y_min_fullres"] * scale)
    
    print(f"Scaled Tile 0: ({x0}, {y0})")
    print(f"Scaled Tile 1: ({x1}, {y1})")
    
    crop0 = np.array(slide.read_region((x0, y0), 0, (224, 224)).convert("RGB"))
    crop1 = np.array(slide.read_region((x1, y1), 0, (224, 224)).convert("RGB"))
    
    print(f"\nCrop 0 RGB Mean: {crop0.mean():.2f}, Std: {crop0.std():.2f}")
    print(f"Crop 1 RGB Mean: {crop1.mean():.2f}, Std: {crop1.std():.2f}")
    print(f"Are crops identical? {np.allclose(crop0, crop1)}")
    
    # Save a visual sanity check
    Image.fromarray(crop0).save("real_tissue_crop0.png")
    Image.fromarray(crop1).save("real_tissue_crop1.png")
    print("Saved real_tissue_crop0.png and real_tissue_crop1.png!")
