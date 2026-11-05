"""src/hne/feature_extraction/conch_v15_model.py"""

from pathlib import Path
import gc
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from hne.core.config import login_huggingface
from hne.core.paths import CONCH_FEATURES, CONCH_PATCH_FEATURES
from hne.feature_extraction.patching import PATCH_SPECS, new_patch_stats, stream_patches_for_tile


def default_conch_v15_transforms(img_size: int = 448):
    """Fallback standard evaluation transform matching CONCH v1.5 pretraining."""
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class ConchV15Extractor:
    """
    CONCH v1.5 Vision-Language Foundation Model Extractor (Mahmood Lab).
    Architecture: ViT-L/16 initialized from UNI, with attentional projection to 768-d.
    Input: 512x512 px crops at 20x magnification (256 um FOV).
    Output: 768-d embedding per patch.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"\nLoading CONCH v1.5 on {self.device}...")

        # Ensure Hugging Face authentication is active for gated weights access
        login_huggingface()

        try:
            from trident.patch_encoder_models import encoder_factory
            self.encoder = encoder_factory(model_name="conch_v15").to(self.device)
            self.encoder.eval()
            self.transform = self.encoder.eval_transforms
            self._use_trident = True
            print("Successfully loaded CONCH v1.5 via TRIDENT encoder_factory.")
        except ImportError:
            # Fallback if trident package is not yet installed in the environment
            import timm
            from huggingface_hub import hf_hub_download

            print("TRIDENT package not found. Attempting direct timm + Hugging Face Hub load...")
            self.encoder = timm.create_model(
                "vit_large_patch16_224",
                img_size=448,
                patch_size=16,
                init_values=1.0,
                num_classes=0,
                dynamic_img_size=True,
            ).to(self.device)
            ckpt_path = hf_hub_download(repo_id="MahmoodLab/conchv1_5", filename="pytorch_model_vision.bin")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder.eval()
            self.transform = default_conch_v15_transforms(img_size=448)
            self._use_trident = False

    def extract_patient_tiles(
        self,
        patient_id: str,
        slide,
        tiles_df,
        fullres_pixel_size: float,
        tile_size_px_fullres: int,
        output_dir: Path = CONCH_PATCH_FEATURES,
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
            batch_size: Mini-batch size for ViT-L/16 inference.
            unpooled: If True, saves unpooled patch embeddings (N_patches, 768) for ABMIL.
                      If False, saves mean-pooled tile embedding (768,) for Ridge baseline.

        Returns:
            List of tile diagnostic dicts (tile_id, stats, embedding_written).
        """
        spec = PATCH_SPECS["conch_v15"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "conch_v15_patch_features" if unpooled else "conch_v15_features"

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

            patch_embeddings = np.concatenate(patch_embeddings, axis=0)  # (N_patches, 768)
            if unpooled:
                # Save full unpooled patch array for Phase 3 ABMIL
                np.save(out_file, patch_embeddings.astype(np.float32))
            else:
                # Average patch vectors to create tile vector for Ridge baseline
                tile_embedding = patch_embeddings.mean(axis=0).astype(np.float32)  # (768,)
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
            output = self.encoder(tensors)
            # If the encoder output is a dict or tuple, extract the embedding tensor
            if isinstance(output, dict):
                feats = output.get("embeddings", output.get("features", next(iter(output.values()))))
            elif isinstance(output, (tuple, list)):
                feats = output[0]
            else:
                feats = output
            return feats.cpu().numpy().astype(np.float32)
