"""src/hne/feature_extraction/virchow2_model.py"""

from pathlib import Path
import gc
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from hne.core.config import login_huggingface
from hne.core.paths import VIRCHOW2_FEATURES, VIRCHOW2_PATCH_FEATURES
from hne.feature_extraction.patching import PATCH_SPECS, new_patch_stats, stream_patches_for_tile


class Virchow2Extractor:
    """
    Virchow2 Foundation Model Feature Extractor.
    Architecture: ViT-H/14 (632M params).
    Input: 224x224 px patches at 20x magnification (112 um FOV).
    Output: 2560-d embedding (concatenation of CLS token [1280-d] and mean-pooled patch tokens [1280-d]).
    """
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"\nLoading Virchow2 on {self.device}...")

        # Ensure Hugging Face authentication is active for gated weights access
        login_huggingface()

        # Load ViT-H/14 backbone with Paige-specific SwiGLU MLP and SiLU activation
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
        ).to(self.device)
        self.model.eval()

        # Input preprocessing (224x224 bicubic resize, ImageNet normalization)
        config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = create_transform(**config)

    def extract_patient_tiles(
        self,
        patient_id: str,
        slide,
        tiles_df,
        fullres_pixel_size: float,
        tile_size_px_fullres: int,
        output_dir: Path = VIRCHOW2_PATCH_FEATURES,
        batch_size: int = 16,
        unpooled: bool = True,
    ):
        """
        Extract sub-patches for all tumor tiles of a patient and save embeddings.

        Args:
            patient_id: Clean patient ID.
            slide: openslide.OpenSlide handle for fullres H&E scan.
            tiles_df: DataFrame with columns tile_id, x_min_fullres, y_min_fullres.
            fullres_pixel_size: Scan resolution in um/px (e.g. ~0.525 um/px).
            tile_size_px_fullres: Dimensions of 1 mm tile in fullres pixels (~1904 px).
            output_dir: Directory where .npy files will be saved.
            batch_size: Mini-batch size for ViT-H/14 inference.
            unpooled: If True, saves unpooled patch embeddings (N_patches, 2560) for ABMIL.
                      If False, saves mean-pooled tile embedding (2560,) for Ridge baseline.

        Returns:
            List of tile diagnostic dicts (tile_id, stats, embedding_written).
        """
        spec = PATCH_SPECS["virchow2"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "virchow2_patch_features" if unpooled else "virchow2_features"

        tile_records = []

        for tile_idx, (_, row) in enumerate(tiles_df.iterrows()):
            tile_id = row["tile_id"]
            out_file = output_dir / f"{patient_id}_{tile_id}_{suffix}.npy"
            if out_file.exists():
                continue

            stats = new_patch_stats()
            x0 = int(round(float(row["x_min_fullres"])))
            y0 = int(round(float(row["y_min_fullres"])))

            patch_generator = stream_patches_for_tile(
                slide=slide,
                x0=x0,
                y0=y0,
                tile_size_px_fullres=tile_size_px_fullres,
                fullres_pixel_size=fullres_pixel_size,
                spec=spec,
                min_tissue_fraction=0.3,
                stats=stats,
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

            # Remaining patches
            if current_batch:
                feats = self._process_batch(current_batch)
                patch_embeddings.append(feats)
                for img in current_batch:
                    img.close()
                current_batch = []

            tile_records.append({
                "patient_id": patient_id,
                "tile_id": tile_id,
                "x0": x0,
                "y0": y0,
                **stats,
                "embedding_written": bool(patch_embeddings),
            })

            if not patch_embeddings:
                continue

            patch_embeddings = np.concatenate(patch_embeddings, axis=0)  # (N_patches, 2560)
            if unpooled:
                # Save full unpooled patch array for Phase 3 ABMIL
                np.save(out_file, patch_embeddings.astype(np.float32))
            else:
                # Average patch vectors to create tile vector for Ridge baseline
                tile_embedding = patch_embeddings.mean(axis=0).astype(np.float32)  # (2560,)
                np.save(out_file, tile_embedding)

            # Prevent GPU/RAM memory bloat
            if tile_idx % 25 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return tile_records

    def _process_batch(self, batch: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
        with torch.inference_mode():
            output = self.model(tensors)  # (B, 261, 1280)
            cls_token = output[:, 0]      # (B, 1280) - Class token
            patch_tokens = output[:, 5:]  # (B, 256, 1280) - Tokens 1-4 are register tokens
            # Standard Virchow2 2560-d representation: [CLS || mean(patch_tokens)]
            feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)  # (B, 2560)
            return feats.cpu().numpy().astype(np.float32)

