"""src/hne/feature_extraction/phikon_v2_model.py"""

from pathlib import Path
import gc
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from hne.core.paths import PHIKON_FEATURES
from hne.feature_extraction.patching import PATCH_SPECS, stream_patches_for_tile


class PhikonV2Extractor:
    def __init__(self):
        # detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading Phikon-v2 on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        self.model = AutoModel.from_pretrained("owkin/phikon-v2").to(self.device)
        self.model.eval()

    def extract_patient_tiles(
        self,
        patient_id: str,
        slide,
        tiles_df,
        fullres_pixel_size: float,
        tile_size_px_fullres: int,
        coord_scale_factor: float = 1.0,
        output_dir: Path = PHIKON_FEATURES,
        batch_size: int = 16, 
    ):
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

        for tile_idx, (_, row) in enumerate(tiles_df.iterrows()):
            tile_id = row["tile_id"]
            out_file = output_dir / f"{patient_id}_{tile_id}_phikon_features.npy"
            if out_file.exists():
                continue

            # scale fullres pixel coordinates to the actual slide canvas
            if "x_min_fullres" in row and "y_min_fullres" in row:
                x0 = int(round(float(row["x_min_fullres"]) * coord_scale_factor))
                y0 = int(round(float(row["y_min_fullres"]) * coord_scale_factor))
            else:
                x0 = int(round(float(row["tile_col"]) * tile_size_px_fullres))
                y0 = int(round(float(row["tile_row"]) * tile_size_px_fullres))


            patch_generator = stream_patches_for_tile(
                slide=slide,
                x0=x0,
                y0=y0,
                tile_size_px_fullres=tile_size_px_fullres,
                fullres_pixel_size=fullres_pixel_size,
                spec=spec,
                min_tissue_fraction=0.3,    # 0.3 allows realistic biopsy edge coverage
            )

            patch_embeddings = []
            current_batch = []

            for patch in patch_generator:
                current_batch.append(patch)
                if len(current_batch) >= batch_size:
                    feats = self._process_batch(current_batch)
                    patch_embeddings.append(feats)
                    for img in current_batch:
                        img.close()
                    current_batch = []

            # remaining patches
            if current_batch:
                feats = self._process_batch(current_batch)
                patch_embeddings.append(feats)
                for img in current_batch:
                    img.close()
                current_batch = []

            if not patch_embeddings:
                continue

            patch_embeddings = np.concatenate(patch_embeddings, axis=0)  # (N_patches, 1024)
            # average all patch vectors to create the tile vector (it can be enhanced basted on the patch cell deconvolution)
            tile_embedding = patch_embeddings.mean(axis=0).astype(np.float32)  # (1024,)

            np.save(out_file, tile_embedding)

            # prevent memory bloat every 50 tiles
            if tile_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def _process_batch(self, batch: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(batch, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
            feats = output.last_hidden_state[:, 0, :].cpu().numpy()
        del inputs, output
        return feats



