import pandas as pd
import numpy as np
from PIL import Image
import torch
import glob
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

from hne.core.paths import PHIKON_FEATURES, TILES

def phikon_feature_extraction(patients_dir: str = TILES,
                              output_dir: str = PHIKON_FEATURES):
    """
    Extracting tiles features using Phikon-v2 foundation model
    """
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients_dir = Path(patients_dir) 
    patient_dirs = [d for d in patients_dir.iterdir() if d.is_dir()]
    
    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        
        # get all the tiles for this one patient
        tile_paths = sorted(patient_dir.glob("*.png"))
        if not tile_paths:
            print(f"No tile images found for {patient_name}")
            continue
        # process each tile        
        for i, tile_path in enumerate(tile_paths):
            image = [Image.open(tile_paths).convert("RGB")]
            # process img
            input = processor(image, return_tensors="pt")
            # get the features
            with torch.inference_mode():
                output = model(**input)
                features = output.last_hidden_state[:, 0, :] # shape: (1, 1024)

            assert features.shape == (1, 1024)

            tile_filename = tile_path.stem
            output_file = output_dir / f"{tile_filename}_phikon_features.npy"
            np.save(output_file, features.numpy())

    print("\nFeature extraction with Phikon-v2 completed!")        
