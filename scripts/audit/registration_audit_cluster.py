"""scripts/audit/registration_audit_cluster.py

Registration audit, part B (cluster, CPU only): for every patient, open the CytAssist image that
feature extraction opened before the fix, and record what it is and where the tiles landed on it.

    python scripts/audit/registration_audit_cluster.py
    python scripts/audit/registration_audit_cluster.py --patients CH_L_275a CH_L_282a
    python scripts/audit/registration_audit_cluster.py --only-he-check   # just verify the full-resolution H&E files (fast)
    python scripts/audit/registration_audit_cluster.py --spot-purity   # colour spots by their own tumor fraction (loads each h5ad; slower)

Resumable: patients already in slide_inventory.csv are skipped. A failing patient is
logged to audit_failures.csv and the run continues.

Outputs (in results/registration_audit/ by default):
    slide_inventory.csv        one row per patient
    overlays/{patient}.jpg     left: the opened slide with spots and tiles where the extractor put them;
                               right: Space Ranger's hires image with the same spots and tiles where they truly are
    sample_crops/{patient}_{tile}.png   patches exactly as fed to Phikon-v2
    bucket_census.csv          every folder of the raw and processed datasets: files, sizes, extensions
    converted_he_check.csv     full-resolution H&E per patient: exists, size; a few opened: dimensions, mpp, match with Space Ranger
    audit_failures.csv         patients that raised, with the error
"""
import argparse
import io
import tempfile
import traceback
from pathlib import Path
import openslide
import pandas as pd
import tifffile
from PIL import Image

from hne.core.config import PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX, RAW_DATA_BUCKET, RAW_DATA_PREFIX, ROOT
from hne.core.data_io import get_s3_loader, load_cytassist_slide, load_spots, load_visium
from hne.core.paths import (CYTASSIST_MAP, HE_MAP, PATIENT_IDS, PATIENTS, PREPROCESSING_QC_REPORTS, REGISTRATION_AUDIT,
                            TILES_SIGNATURE_MATRIX)
from hne.feature_extraction.registration_audit import audit_patient
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG
from hne.preprocessing.tumor_purity import attach_tumor_fraction

TILE_COLUMNS = ["tile_id", "tile_row", "tile_col", "tile_purity",
                "x_min_fullres", "y_min_fullres", "x_max_fullres", "y_max_fullres"]


def append_row(row: dict, path: Path):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def audit_one(patient_id: str, metadata: pd.DataFrame, output_dir: Path, spot_purity: bool) -> dict:
    loader = get_s3_loader()
    paths = PATIENTS[patient_id]
    # the image the audited extraction opened: the CytAssist image
    filename = CYTASSIST_MAP.get(paths.clean_id)
    if not filename:
        raise FileNotFoundError(f"{patient_id}: no CytAssist image in the manifest")

    bucket, key = loader._parse_s3_path(f"{paths.cytassist_image_prefix}/{filename}")
    tif_size = loader.s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"]

    # scale factors: an unreadable file is recorded as the fallback the old extractor took
    scale_json, scale_json_error = None, ""
    try:
        scale_json = loader.read_json(f"{paths.visium_info}/scalefactors_json.json")
    except Exception as e:
        scale_json_error = repr(e)

    spots = load_spots(paths)
    if spot_purity:
        merged, _ = attach_tumor_fraction(spots, load_visium(paths), patient_id=patient_id)
        if merged is not None:
            spots = merged

    tiles_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv"
    tiles_df = pd.read_csv(tiles_csv) if tiles_csv.exists() else pd.DataFrame(columns=TILE_COLUMNS)

    # pixel size and tile size as preprocessing computed them; recomputed for patients it excluded
    if patient_id in metadata.index and pd.notna(metadata.loc[patient_id, "fullres_pixel_size"]):
        fullres_px = float(metadata.loc[patient_id, "fullres_pixel_size"])
        tile_px = int(metadata.loc[patient_id, "tile_size_pixels"])
    elif scale_json and "spot_diameter_fullres" in scale_json:
        fullres_px = PREPROCESSING_CONFIG.spot_diameter_um / float(scale_json["spot_diameter_fullres"])
        tile_px = int(PREPROCESSING_CONFIG.target_physical_size_um / fullres_px)
    else:
        raise ValueError(f"{patient_id}: no pixel size in preprocessing metadata and no spot_diameter_fullres")

    hires_image = None
    try:
        hires_image = Image.open(io.BytesIO(loader._read_bytes(f"{paths.visium_info}/tissue_hires_image.png")))
        hires_image.load()
    except Exception:
        pass   # the reference panel is optional

    with load_cytassist_slide(paths) as slide:
        row = audit_patient(
            patient_id=patient_id, slide=slide, scale_json=scale_json, spots=spots, tiles_df=tiles_df,
            fullres_pixel_size=fullres_px, tile_size_px=tile_px, output_dir=output_dir, hires_image=hires_image,
        )
    return {"patient_id": patient_id, "tif_filename": filename, "tif_size_bytes": tif_size,
            "has_tile_matrix": tiles_csv.exists(), "scale_json_error": scale_json_error,
            "hires_reference_found": hires_image is not None, **row}


def folder_census(bucket: str, prefix: str, label: str, max_depth: int = 5) -> list[dict]:
    """
    Walk the folders under a prefix (not into .zarr stores) and summarise the files directly in
    each: count, total size, extensions, largest file. Shows what images exist and where.
    """
    loader = get_s3_loader()
    paginator = loader.s3_client.get_paginator("list_objects_v2")
    rows, queue = [], [(f"{prefix.rstrip('/')}/", 0)]
    while queue:
        folder, depth = queue.pop(0)
        n_files, total, largest, largest_key, extensions = 0, 0, 0, "", {}
        for page in paginator.paginate(Bucket=bucket, Prefix=folder, Delimiter="/"):
            for obj in page.get("Contents", []):
                n_files += 1
                total += obj["Size"]
                ext = Path(obj["Key"]).suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                if obj["Size"] > largest:
                    largest, largest_key = obj["Size"], obj["Key"]
            for sub in page.get("CommonPrefixes", []):
                if depth < max_depth and ".zarr" not in sub["Prefix"]:
                    queue.append((sub["Prefix"], depth + 1))
        if n_files:
            rows.append({"dataset": label, "folder": folder, "n_files": n_files, "total_gb": round(total / 1024 ** 3, 2),
                         "largest_file_mb": round(largest / 1024 ** 2, 1), "largest_file": largest_key.rsplit("/", 1)[-1],
                         "extensions": " ".join(f"{k or '(none)'}:{v}" for k, v in sorted(extensions.items()))})
    return rows


def locate_fullres_he(output_dir: Path, verify_n: int):
    """
    Find the full-resolution H&E. The MOSAIC Visium pipeline tiled `converted_he/{sample}_vis.tif`
    (see spot_tiling_reference.csv in any image_features folder); this looks for that folder and for
    any other folder of large images, in both the raw and the processed dataset.
    """
    rows = (folder_census(RAW_DATA_BUCKET, RAW_DATA_PREFIX, "raw") +
            folder_census(PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX, "processed"))
    census = pd.DataFrame(rows)
    census.to_csv(output_dir / "bucket_census.csv", index=False)

    images = census[census["largest_file_mb"] >= 100]
    print(f"\nFolders holding a file of 100 MB or more ({len(images)} of {len(census)} folders):")
    if not images.empty:
        print(images[["dataset", "folder", "n_files", "largest_file_mb", "largest_file", "extensions"]].to_string(index=False))
    he = census[census["folder"].str.contains("converted_he|image_features", regex=True)]
    print("\nFolders named converted_he or image_features:")
    print(he[["dataset", "folder", "n_files", "largest_file_mb"]].head(60).to_string(index=False) if not he.empty else "  none found")

    check_fullres_he(output_dir, verify_n)


def check_fullres_he(output_dir: Path, verify_n: int):
    """
    For every patient: does the full-resolution H&E exist under PatientS3Paths.he_image_prefix, and how big is it?
    For the first `verify_n` patients with tiles: download it, open it, and compare its size with the
    frame Space Ranger's coordinates imply.
    """
    loader = get_s3_loader()
    inventory = pd.read_csv(output_dir / "slide_inventory.csv").set_index("patient_id") if (output_dir / "slide_inventory.csv").exists() else pd.DataFrame()
    to_open = [p for p in PATIENT_IDS if (TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{p}.csv").exists()][:max(verify_n, 0)]
    checks = []
    for patient_id in PATIENT_IDS:
        # the manifest's filename once the inventory has been rerun, else the pipeline's naming
        filename = HE_MAP.get(patient_id, f"{patient_id}_vis.tif")
        bucket, key = loader._parse_s3_path(f"{PATIENTS[patient_id].he_image_prefix}/{filename}")
        record = {"patient_id": patient_id, "key": key, "exists": False}
        try:
            record["size_mb"] = round(loader.s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"] / 1024 ** 2, 1)
            record["exists"] = True
        except Exception as e:
            record["error"] = repr(e)
        if record["exists"] and patient_id in to_open:
            try:
                with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                    loader.s3_client.download_file(bucket, key, tmp.name)
                    try:
                        slide = openslide.OpenSlide(tmp.name)
                        record.update(opens_with_openslide=True, width=slide.dimensions[0], height=slide.dimensions[1],
                                      levels=slide.level_count, mpp_x=slide.properties.get("openslide.mpp-x"))
                        slide.close()
                    except Exception as e:
                        with tifffile.TiffFile(tmp.name) as tif:
                            page = tif.pages[0]
                            record.update(opens_with_openslide=False, openslide_error=repr(e), width=page.imagewidth,
                                          height=page.imagelength, levels=len(tif.series[0].levels),
                                          tiff_resolution=str(page.tags["XResolution"].value) if "XResolution" in page.tags else None)
                if patient_id in inventory.index and pd.notna(inventory.loc[patient_id, "implied_fullres_width"]):
                    implied = float(inventory.loc[patient_id, "implied_fullres_width"])
                    record["implied_fullres_width"] = implied
                    record["matches_fullres_frame"] = abs(record["width"] - implied) / implied < 0.01
            except Exception as e:
                record["error"] = repr(e)
            print(f"  opened {patient_id}: {record}", flush=True)
        checks.append(record)

    df = pd.DataFrame(checks)
    df.to_csv(output_dir / "converted_he_check.csv", index=False)
    print(f"\nFull-resolution H&E found for {int(df['exists'].sum())} of {len(df)} patients "
          f"(median {df['size_mb'].median():.0f} MB)." if df["exists"].any() else "\nNo full-resolution H&E found at the expected path.")
    missing = df.loc[~df["exists"], "patient_id"].tolist()
    if missing:
        print(f"Missing: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", nargs="+", default=None, help="patient IDs (default: all in the manifest)")
    parser.add_argument("--output-dir", type=Path, default=REGISTRATION_AUDIT)
    parser.add_argument("--spot-purity", action="store_true",
                        help="colour spots by their own tumor fraction instead of their tile's purity")
    parser.add_argument("--skip-s3-listing", action="store_true", help="do not survey the buckets for the full-resolution H&E")
    parser.add_argument("--only-he-check", action="store_true",
                        help="skip the per-patient audit and the bucket survey; only check the full-resolution H&E files")
    parser.add_argument("--verify-he", type=int, default=3,
                        help="number of converted_he slides to download and open (0 to skip)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.only_he_check:
        check_fullres_he(output_dir, args.verify_he)
        print(f"\nTo send the result back:\n  git add {output_dir / 'converted_he_check.csv'}\n"
              '  git commit -m "chore(audit): full-resolution H&E check"\n  git push')
        return
    inventory_csv = output_dir / "slide_inventory.csv"
    failures_csv = output_dir / "audit_failures.csv"

    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv").set_index("patient_id")
    patients = args.patients or PATIENT_IDS
    done = set(pd.read_csv(inventory_csv)["patient_id"]) if inventory_csv.exists() else set()

    n_ok = n_failed = 0
    for i, patient_id in enumerate(patients, start=1):
        if patient_id in done:
            continue
        try:
            row = audit_one(patient_id, metadata, output_dir, args.spot_purity)
            append_row(row, inventory_csv)
            n_ok += 1
            print(f"[{i}/{len(patients)}] {patient_id}: slide {row['slide_width']}x{row['slide_height']} | "
                  f"{row['effective_um_per_px']:.2f} um/px | {row['n_tiles_partly_outside_slide']}/{row['n_tiles']} "
                  f"tiles outside | matches regist target: {row['slide_matches_regist_target']}", flush=True)
        except Exception as e:
            n_failed += 1
            append_row({"patient_id": patient_id, "error": repr(e), "traceback": traceback.format_exc()}, failures_csv)
            print(f"[{i}/{len(patients)}] {patient_id}: FAILED {e!r}", flush=True)

    print(f"\nAudited {n_ok} patients, {n_failed} failed, {len(done & set(patients))} already done.")

    if not args.skip_s3_listing:
        try:
            locate_fullres_he(output_dir, args.verify_he)
        except Exception as e:
            print(f"Locating the full-resolution H&E failed: {e!r}")
            traceback.print_exc()

    if inventory_csv.exists():
        inv = pd.read_csv(inventory_csv)
        print(f"\nSlides by size:\n{inv.groupby(['slide_width', 'slide_height']).size().to_string()}")
        print(f"Effective um/px: median {inv['effective_um_per_px'].median():.2f}, "
              f"range {inv['effective_um_per_px'].min():.2f} to {inv['effective_um_per_px'].max():.2f}")
        print(f"Patients with a scale fallback: {int(inv['fallback_hit'].sum())}")
        print(f"Tiles partly outside the slide: {int(inv['n_tiles_partly_outside_slide'].sum())} of {int(inv['n_tiles'].sum())}")

    rel = output_dir.resolve().relative_to(ROOT) if output_dir.resolve().is_relative_to(ROOT) else output_dir
    print("\nTo send the results back, run:")
    print(f"  git add {rel}")
    print('  git commit -m "chore(audit): registration audit results from the cluster"')
    print("  git push")


if __name__ == "__main__":
    main()
