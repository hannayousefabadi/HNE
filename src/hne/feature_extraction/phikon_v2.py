import pandas as pd
import numpy as np
from PIL import Image
import torch
import glob
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

from hne.core.paths import PHIKON_FEATURES, TILES

class PhikonExtractor:
    def __init__(self):
        """
        To initialize the model once
        """
        # to detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading Phikon-v2 on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        self.model.to(self.device) # move model to GPU/CPU
        self.model.eval()

    def extract_patient(self, patients_dir: str = TILES,
                                output_dir: str = PHIKON_FEATURES):
        """
        Extracting tiles features using Phikon-v2 foundation model
        """
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
            for tile_path in tile_paths:
                image = [Image.open(tile_path).convert("RGB")]
                # process img
                inputs = self.processor(image, return_tensors="pt")
                # move inputs to the same device as the model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # get the features
                with torch.inference_mode():
                    output = self.model(**inputs)
                    features = output.last_hidden_state[:, 0, :] # shape: (1, 1024)

                assert features.shape == (1, 1024)

                tile_filename = tile_path.stem
                output_file = output_dir / f"{tile_filename}_phikon_features.npy"
                # move items back to CPU before saving as np
                np.save(output_file, features.cpu().numpy())

        print("\nFeature extraction with Phikon-v2 completed!")        
