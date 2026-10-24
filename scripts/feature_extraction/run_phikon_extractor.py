"""scripts/feature_extraction/run_phikon_extractor.py

Phikon-v2 feature extraction for the cohort.

Supports two modes:
  1. Default (Phase 1): Mean-pooled tile embeddings (1024,) saved to feature_sets/phikon_v2_features/
  2. Unpooled (Phase 2): Raw patch embeddings (N_patches, 1024) saved to feature_sets/phikon_v2_patch_features/
     Run via: python scripts/feature_extraction/run_phikon_extractor.py --unpooled

Patients are processed in small groups, each group in a fresh worker process. When a worker
exits, all of its memory (RAM and GPU) goes back to the system, so nothing can build up over
the cohort. If a worker dies, the parent reports how, retries its patients one at a time, and
carries on. Rerunning the script resumes: finished tiles are skipped.
"""
import argparse
from datetime import datetime
import gc
import glob
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional
import pandas as pd

from hne.core.paths import (PATIENTS, PATIENT_IDS, TILES_SIGNATURE_MATRIX,
                            PREPROCESSING_QC_REPORTS, PHIKON_FEATURES, PHIKON_PATCH_FEATURES,
                            FEATURE_EXTRACTION_QC_REPORTS)
from hne.core.s3_io import remove_orphan_slides, temp_dir_usage
from hne.utils import memory_snapshot

CONFIG = {
    "patients_per_worker": 4,   # a fresh process (and model load) every this many patients
    "batch_size": 16,
}


def get_log_paths(unpooled: bool = False) -> dict:
    sub = "phikon_v2_unpooled" if unpooled else "phikon_v2"
    log_dir = FEATURE_EXTRACTION_QC_REPORTS / sub
    return {
        "log_dir": log_dir,
        "patient_log": log_dir / "patient_log.csv",
        "tile_log": log_dir / "tile_log.csv",
        "failure_log": log_dir / "failures.csv",
        "run_log": log_dir / "extraction.log",
    }


def log(message: str, run_log: Optional[Path] = None):
    """Timestamped line to the console and to extraction.log, flushed at once so it survives a kill."""
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} [pid {os.getpid()}] {message}"
    print(line, flush=True)
    if run_log is not None:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with open(run_log, "a") as f:
            f.write(line + "\n")


def append_rows(rows: list[dict], path: Path):
    """
    Append records to a CSV log. If the file was written with other columns (an older version
    of this script), it is rewritten with the union of both, so the file always parses.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False)
        return
    existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
    if existing_columns == new.columns.tolist():
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(path), new], ignore_index=True).to_csv(path, index=False)


def resources() -> dict:
    """Memory and temp-space readings, logged per patient to show what grows."""
    import torch
    snap = {**memory_snapshot(), **temp_dir_usage()}
    snap["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else None
    return snap


def patient_tiles_csv(patient_id: str) -> Path:
    return TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{patient_id}.csv"


def pending_patients(output_dir: Path, tile_log: Path, unpooled: bool = False) -> list[str]:
    """
    Patients that still have a tile to extract. A tile is done when its feature file exists or
    when the tile log records it (a tile with no tissue is logged and gets no file).
    """
    logged = set()
    if tile_log.exists():
        df_log = pd.read_csv(tile_log, usecols=["patient_id", "tile_id"])
        logged = set(zip(df_log["patient_id"], df_log["tile_id"].astype(str)))

    suffix = "phikon_patch_features" if unpooled else "phikon_features"
    pending = []
    for patient_id in PATIENT_IDS:
        csv_path = patient_tiles_csv(patient_id)
        if not csv_path.exists():
            continue
        tile_ids = pd.read_csv(csv_path, usecols=["tile_id"])["tile_id"].astype(str)
        if any((patient_id, tile_id) not in logged
               and not (output_dir / f"{patient_id}_{tile_id}_{suffix}.npy").exists()
               for tile_id in tile_ids):
            pending.append(patient_id)
    return pending


def extract_patient(
    phikon,
    patient_id: str,
    metadata: pd.DataFrame,
    output_dir: Path,
    tile_log_path: Path,
    patient_log_path: Path,
    unpooled: bool = False,
):
    """Extract every remaining tile of one patient and write its log rows."""
    from hne.core.data_io import load_he_slide, slide_um_per_px
    from hne.feature_extraction.patching import require_fullres_slide

    patient_tiles = pd.read_csv(patient_tiles_csv(patient_id))
    patient_meta = metadata[metadata["patient_id"] == patient_id]
    if patient_meta.empty:
        raise ValueError(f"{patient_id}: has a tile matrix but no row in the preprocessing metadata.csv")

    fullres_px_size = float(patient_meta["fullres_pixel_size"].iloc[0])
    tile_size_px = int(patient_meta["tile_size_pixels"].iloc[0])

    with load_he_slide(PATIENTS[patient_id]) as slide:
        if slide is None:
            raise FileNotFoundError(f"{patient_id}: no H&E scan in the manifest")

        # the slide must be the full-resolution H&E: tile coordinates are used as they are
        slide_w, slide_h = slide.dimensions
        require_fullres_slide(patient_id, slide_w, slide_h, patient_tiles)

        # tiles were laid out by preprocessing with a pixel size; it must be this scan's
        slide_px_size = slide_um_per_px(slide, patient_id)
        if abs(slide_px_size - fullres_px_size) / slide_px_size > 0.01:
            raise ValueError(
                f"{patient_id}: preprocessing used {fullres_px_size:.4f} um/px but the H&E scan is "
                f"{slide_px_size:.4f} um/px. The tile matrix is from an older preprocessing run; rerun preprocessing."
            )

        tile_records = phikon.extract_patient_tiles(
            patient_id=patient_id,
            slide=slide,
            tiles_df=patient_tiles,
            fullres_pixel_size=fullres_px_size,
            tile_size_px_fullres=tile_size_px,
            output_dir=output_dir,
            batch_size=CONFIG["batch_size"],
            unpooled=unpooled,
        )

    # extraction log: what was read for this patient, and from where
    df_tile_log = pd.DataFrame(tile_records)
    n_oob = int(df_tile_log["n_out_of_bounds"].sum()) if not df_tile_log.empty else 0
    append_rows(tile_records, tile_log_path)
    append_rows([{
        "patient_id": patient_id,
        "slide_width": slide_w,
        "slide_height": slide_h,
        "um_per_px": fullres_px_size,
        "tile_size_px": tile_size_px,
        "unpooled": unpooled,
        "n_tiles_processed": len(df_tile_log),
        "n_tiles_written": int(df_tile_log["embedding_written"].sum()) if not df_tile_log.empty else 0,
        "n_tiles_with_out_of_bounds": int((df_tile_log["n_out_of_bounds"] > 0).sum()) if not df_tile_log.empty else 0,
        "n_patches_out_of_bounds": n_oob,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **resources(),
    }], patient_log_path)
    return len(df_tile_log), n_oob


def worker(patient_ids: list[str], unpooled: bool, output_dir: Path, log_paths: dict):
    """Runs in its own process: load the model once, extract these patients, exit."""
    import torch
    from hne.feature_extraction.phikon_v2_model import PhikonV2Extractor

    phikon = PhikonV2Extractor()
    metadata = pd.read_csv(PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")
    log(f"worker started for {patient_ids} | unpooled={unpooled} | {resources()}", log_paths["run_log"])

    for patient_id in patient_ids:
        try:
            n_tiles, n_oob = extract_patient(
                phikon=phikon,
                patient_id=patient_id,
                metadata=metadata,
                output_dir=output_dir,
                tile_log_path=log_paths["tile_log"],
                patient_log_path=log_paths["patient_log"],
                unpooled=unpooled,
            )
            log(f"{patient_id}: {n_tiles} tiles extracted, {n_oob} patches out of bounds | {resources()}", log_paths["run_log"])
        except Exception as e:
            # one patient's error must not stop the others; it is recorded and reported at the end
            append_rows([{"patient_id": patient_id, "kind": "error", "detail": repr(e),
                          "traceback": traceback.format_exc()}], log_paths["failure_log"])
            log(f"{patient_id}: FAILED {e!r}", log_paths["run_log"])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def describe_exit(exitcode: int) -> str:
    if exitcode == -9:
        return "killed by signal 9 (SIGKILL): this is what the kernel's out-of-memory killer does"
    if exitcode is not None and exitcode < 0:
        return f"killed by signal {-exitcode}"
    return f"exit code {exitcode}"


def run_worker(context, patient_ids: list[str], unpooled: bool, output_dir: Path, log_paths: dict) -> int:
    process = context.Process(target=worker, args=(patient_ids, unpooled, output_dir, log_paths))
    process.start()
    process.join()
    return process.exitcode


def extract_features(unpooled: bool = False, output_dir: Optional[Path] = None):
    output_dir = Path(output_dir) if output_dir else (PHIKON_PATCH_FEATURES if unpooled else PHIKON_FEATURES)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = get_log_paths(unpooled=unpooled)
    log_paths["log_dir"].mkdir(parents=True, exist_ok=True)

    mode_str = "UNPOOLED (Phase 2 ABMIL patch arrays)" if unpooled else "MEAN-POOLED (Phase 1 tile vectors)"
    log(f"Starting Phikon-v2 Feature Extraction in {mode_str} mode", log_paths["run_log"])
    log(f"Output directory: {output_dir}", log_paths["run_log"])

    # slides left behind by killed runs take space
    removed, freed_gb = remove_orphan_slides()
    log(f"run started | removed {removed} orphan slide file(s), {freed_gb:.1f} GB | {memory_snapshot()} | {temp_dir_usage()}", log_paths["run_log"])
    older = [f for f in glob.glob(os.path.join(tempfile.gettempdir(), "tmp*.tif")) if os.path.isfile(f)]
    if older:
        size_gb = sum(os.path.getsize(f) for f in older) / 1024 ** 3
        log(f"WARNING: {len(older)} file(s) matching {tempfile.gettempdir()}/tmp*.tif ({size_gb:.1f} GB) look like slides left by "
            f"earlier runs. They are not deleted automatically; remove them if they are not in use.", log_paths["run_log"])
    if temp_dir_usage()["temp_dir_in_ram"]:
        log("WARNING: the temp directory is RAM-backed. Every downloaded slide counts as memory. "
            "Set TMPDIR to a directory on disk before running.", log_paths["run_log"])

    pending = pending_patients(output_dir=output_dir, tile_log=log_paths["tile_log"], unpooled=unpooled)
    log(f"{len(pending)} patient(s) to extract", log_paths["run_log"])

    context = mp.get_context("spawn")    # a clean interpreter per worker; required with CUDA
    size = CONFIG["patients_per_worker"]
    died = []
    for start in range(0, len(pending), size):
        group = pending[start:start + size]
        exitcode = run_worker(context, group, unpooled, output_dir, log_paths)
        if exitcode == 0:
            continue

        # the worker died without finishing: find out which patient, one process each
        log(f"WORKER DIED on {group}: {describe_exit(exitcode)} | {memory_snapshot()}", log_paths["run_log"])
        remove_orphan_slides()
        still_pending = set(pending_patients(output_dir=output_dir, tile_log=log_paths["tile_log"], unpooled=unpooled))
        for patient_id in [p for p in group if p in still_pending]:
            exitcode = run_worker(context, [patient_id], unpooled, output_dir, log_paths)
            if exitcode != 0:
                log(f"WORKER DIED on {patient_id} alone: {describe_exit(exitcode)} | {memory_snapshot()}", log_paths["run_log"])
                append_rows([{"patient_id": patient_id, "kind": "worker_died", "detail": describe_exit(exitcode),
                              "traceback": ""}], log_paths["failure_log"])
                died.append(patient_id)
                remove_orphan_slides()

    left = pending_patients(output_dir=output_dir, tile_log=log_paths["tile_log"], unpooled=unpooled)
    suffix = "phikon_patch_features" if unpooled else "phikon_features"
    n_features = len(list(output_dir.glob(f"*_{suffix}.npy")))
    log(f"run finished | {n_features} feature files in {output_dir} | {len(left)} patient(s) still incomplete: {left}", log_paths["run_log"])
    if left:
        log(f"See {log_paths['failure_log']} for the reason of each failure.", log_paths["run_log"])
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phikon-v2 Feature Extractor")
    parser.add_argument(
        "--unpooled",
        action="store_true",
        help="Extract unpooled patch embeddings (M_patches x 1024) for Phase 2 ABMIL instead of mean pooling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory for feature files",
    )
    args = parser.parse_args()
    extract_features(unpooled=args.unpooled, output_dir=args.output_dir)
