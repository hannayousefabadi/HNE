"""scripts/audit/registration_audit_local.py

Registration audit, part A3 (local): for the few patients whose Space Ranger `spatial/`
folder is in data/, run the same audit as the cluster script on the local
`cytassist_image.tiff`. This is valid only where that file is the one extraction read,
which is checked by comparing its byte size with the manifest TIFF's size on S3.
"""
import json
import pandas as pd
from PIL import Image

from hne.core.config import ROOT, RESULTS
from hne.core.paths import PATIENT_IDS, PREPROCESSING_QC_REPORTS, REGISTRATION_AUDIT, TILES_SIGNATURE_MATRIX
from hne.feature_extraction.registration_audit import audit_patient

CONFIG = {
    "data_dir": ROOT / "data",
    "output_dir": REGISTRATION_AUDIT / "local_check",
}


class PILSlide:
    """Minimal OpenSlide-like wrapper around an ordinary image file."""
    level_count = 1
    properties = {}

    def __init__(self, path):
        self.image = Image.open(path).convert("RGB")
        self.dimensions = self.image.size

    def read_region(self, location, level, size):
        x, y = location
        return self.image.crop((x, y, x + size[0], y + size[1]))


def run():
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv").set_index("patient_id")
    tif_report = pd.read_csv(RESULTS / "cohort_metadata" / "tif_diagnostic_report.csv").set_index("patient_id")

    rows = []
    for patient_id in PATIENT_IDS:
        spatial = CONFIG["data_dir"] / patient_id.rstrip("abcdefgh") / "spaceranger_count" / "spatial"
        tiles_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv"
        local_tif = spatial / "cytassist_image.tiff"
        if not (local_tif.exists() and tiles_csv.exists() and patient_id in metadata.index):
            continue

        s3_size = int(tif_report.loc[patient_id, "size_bytes"])
        local_size = local_tif.stat().st_size
        if local_size != s3_size:
            print(f"{patient_id}: local cytassist_image.tiff ({local_size} bytes) differs from the manifest TIFF "
                  f"({s3_size} bytes); skipped")
            continue

        with open(spatial / "scalefactors_json.json") as f:
            scale_json = json.load(f)
        row = audit_patient(
            patient_id=patient_id,
            slide=PILSlide(local_tif),
            scale_json=scale_json,
            spots=pd.read_csv(spatial / "tissue_positions.csv"),
            tiles_df=pd.read_csv(tiles_csv),
            fullres_pixel_size=float(metadata.loc[patient_id, "fullres_pixel_size"]),
            tile_size_px=int(metadata.loc[patient_id, "tile_size_pixels"]),
            output_dir=output_dir,
            hires_image=Image.open(spatial / "tissue_hires_image.png"),
        )
        row = {"patient_id": patient_id, "tif_filename": tif_report.loc[patient_id, "filename"],
               "tif_size_bytes": s3_size, "local_file_same_size_as_s3_tif": True, **row}
        rows.append(row)
        print(f"{patient_id}: slide {row['slide_width']}x{row['slide_height']} | scale {row['scale_used']:.4f} | "
              f"{row['effective_um_per_px']:.2f} um/px | patch {row['patch_px_native']} px native | "
              f"{row['n_patch_windows_clamped']}/{row['n_patch_windows']} windows clamped | "
              f"{row['pct_spots_outside_slide']:.0f}% spots outside | "
              f"regist target should be {row['implied_regist_target_width']:.0f} px wide")

    if not rows:
        raise RuntimeError("No patient has both a local spatial/ folder and a manifest TIFF of the same size.")
    pd.DataFrame(rows).to_csv(output_dir / "slide_inventory.csv", index=False)
    print(f"\n{len(rows)} patients audited locally. Written to {output_dir}")


if __name__ == "__main__":
    run()
