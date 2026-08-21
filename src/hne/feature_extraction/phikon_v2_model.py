"""
src/hne/feature_extraction/phikon_v2_model.py
"""
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

from hne.core.paths import PHIKON_FEATURES
from hne.feature_extraction.patching import PATCH_SPECS, extract_patches_for_tile

class PhikonV2Extractor:
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

    def extract_patient_tiles(self,
                              patient_id: str,
                              slide,
                              tiles_df,
                              fullres_pixel_size: float,
                              tile_size_px_fullres: int,
                              output_dir: Path = PHIKON_FEATURES,
                              batch_size: int = 64):
        """
        Turning patient tiles to model-requirement patches ready for feature extraction
        
        Args:
            tiles_df: patient's tumor tiles (from cohort preprocessing step), must have
            tile_row, tile_col, tile_id.
            slide: openslide.OpenSlide handle for this patient's fullres tiff.
        """
        spec = PATCH_SPECS["phikon_v2"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tiles_df.iterrows():
            tile_id = row["tile_id"]
            x0 = int(row["tile_col"]) * tile_size_px_fullres
            y0 = int(row["tile_row"]) * tile_size_px_fullres

            patches = extract_patches_for_tile(
                slide=slide, x0=x0, y0=y0,
                tile_size_px_fullres=tile_size_px_fullres,
                fullres_pixel_size=fullres_pixel_size,
                spec=spec
            )

            if not patches:
                continue

            images = [p["image"] for p in patches]

            patch_embeddings = []

            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.processor(batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    output = self.model(**inputs)
                    feats = output.last_hidden_state[:, 0, :]

                patch_embeddings.append(feats.cpu().numpy())

            patch_embeddings = np.concatenate(patch_embeddings, axis=0) # (N_patches, 1024)
            # average all patch vectors to create the tile vector (it can be enhanced basted on the patch cell deconvolution)
            tile_embedding = patch_embeddings.mean(axis=0) # (1024,)   

            out_file = output_dir / f"{patient_id}_{tile_id}_phikon_features.npy"
            np.save(out_file, tile_embedding)      

