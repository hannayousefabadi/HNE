# HNE

Predict tumor pathway activity in lung cancer directly from H&E histology. Matched Visium spatial transcriptomics supplies the ground truth: each slide is cut into ~1 mm² tumor tiles, each tile gets a score for 5 pathway signatures (FMRP, Cell_cycle, YAP, WNT, EMT), and a model learns to predict those scores from foundation-model embeddings of the tile image.

## Pipeline

Each stage reads the previous stage's outputs from disk.

| Stage | Run | Logic | Output |
|---|---|---|---|
| Cohort inventory | `scripts/cohort_inventory/cohort_inventory.py` | — | `results/cohort_metadata/cohort_manifest.json` (patient IDs, H&E map, CytAssist map) |
| Preprocessing | `scripts/preprocessing/preprocess_cohort.py` | `src/hne/preprocessing/`, `src/hne/preprocessing_qc/` | `tiles/`, `tile_signature_matrix/`, `qc_reports/preprocessing_qc/` |
| Feature extraction | `scripts/feature_extraction/run_phikon_extractor.py` | `src/hne/feature_extraction/` | `feature_sets/phikon_v2_features/*.npy` |
| Phase 1 baseline | `scripts/model_train/phase1/train_phase1_ridge.py` | `src/hne/models/` | `results/phase1_ridge/` |
| Registration audit | `scripts/audit/` | `src/hne/feature_extraction/registration_audit.py` | `results/registration_audit/` |
| Early Phase 1 MLP | `scripts/model_train/phase1/train_phase1.py` | `src/hne/models/` | `results/phase1/` |

- **Preprocessing**: tumor fraction from deconvolution, Bayesian tile purity, tumor-tile filter, per-spot ssGSEA scores, tile-level means, top-quartile binary calls. Every patient gets a QC verdict (`OK` / `REVIEW` / `EXCLUDE`).
- **Feature extraction**: each tile is split into 112 µm sub-patches, embedded with Phikon-v2 (1024-d CLS token), and mean-pooled to one vector per tile. Runs in a fresh worker process per group of patients and resumes per tile.
- **Phase 1 baseline**: ridge regression per signature, evaluated with nested grouped (by patient) cross-validation and patient-level statistics (`src/hne/models/ridge_cv.py`, `src/hne/models/evaluation.py`).
- **Early Phase 1 MLP**: a distributional MLP with a single train/validation split. Its results are superseded and kept only for comparison.

## Conventions

- `src/hne/` holds importable logic; `scripts/` holds the entry points you actually run. Scripts take their settings from a `CONFIG` dict at the top of the file, not from CLI flags.
- Paths come from `src/hne/core/paths.py` and `src/hne/core/config.py`. Do not hardcode output directories.
- Preprocessing thresholds live in one place: `PREPROCESSING_CONFIG` in `src/hne/preprocessing/preprocessing_config.py`.
- Raw data is read from S3 through `S3DataLoader` (`src/hne/core/s3_io.py`). Bucket names and prefixes come from `.env`.
- `tests/` contains ad-hoc debug scripts, not a test suite.

## Things that bite

- **Feature extraction is incomplete and the Phase 1 results are stale.** `tile_signature_matrix/` is current (full-resolution H&E, 1 mm tiles). `feature_sets/phikon_v2_features/` covers 58 of 91 patients. `results/phase1_ridge/` and `results/phase1/` were computed on the old CytAssist features: do not interpret them. Phase 1 is rerun, unchanged, once extraction is complete (`review/06_registration_audit.md`).
- Images come from `load_he_slide()` only (`{processed}/{PROCESSED_VERSION}/converted_he/`, filenames in the manifest's `he_map`). `load_cytassist_slide()` and `cytassist_image_prefix` are for the audit; never crop tiles or patches from a CytAssist image.
- The H&E image and the Space Ranger output must come from the same pipeline version folder. `PROCESSED_VERSION` in `src/hne/core/config.py` is the one place it is set.
- Fullres coordinates are never rescaled to fit an image. If tiles do not fit inside the opened slide, the image is wrong.
- Physical scale comes from the H&E scan's own metadata (`slide_um_per_px`), nowhere else. Never derive pixel size from `spot_diameter_fullres`, and never use a Space Ranger scale factor to place tiles or patches.
- A tile is `tile_id = "{row}-{col}"` in CSVs and feature filenames, but its PNG is named `tile_r{row}_c{col}.png`.
- A signature that could not be scored is `NaN`, never `0.0`. The Phase 1 loader drops any tile with a `NaN` target.
- Splits and CV folds are by `patient_id`. Never split by tile. Feature and target scaling statistics come from training folds only.
- Statistics count patients, not tiles: CIs from a patient-level bootstrap, p-values from patient-level permutation. Never report a tile-level p-value.
- Data that influenced a choice (epoch, hyperparameter, model) cannot report that choice's result. Do not edit a `CONFIG` and rerun to improve a reported number.
- The early MLP outputs a standard deviation, but no loss trains it. Do not treat `std_pred` as meaningful.
- Scripts fail loudly on a missing input. No fallback values.
- Patients lost in preprocessing are the `EXCLUDE` verdicts in `qc_summary.csv` plus the `Failed patients` list in the cohort log. Patients that crash never reach the QC summary.

## After changing code

- Apply the `update-docs` skill: bring the affected `review/` note, and `README.md` or this file if needed, in line with the change.
- Apply the `update-graphify` skill: run `graphify update .` to refresh the code graph in `graphify-out/`.

## Deeper notes

`review/` has one design-notes file per pipeline stage (decisions, thresholds, alternatives, future plans). Read the relevant one before changing a stage. The directory exists only locally.

`review/` and `graphify-out/` must stay in `.gitignore`. Never commit either, and never remove them from it.

- `review/00_preprocessing_pipeline.md`
- `review/01_feature_extraction_pipeline.md`
- `review/02_early_phase1_modeling_pipeline.md`
- `review/04_phase1_ridge_baseline.md`
- `review/06_registration_audit.md`
- `review/plan_decisions.md` (decisions and evaluation rules for all phases)
- `review/00_glossary.md` (project vocabulary)
- `review/Privé et partagé/` (export of the MOSAIC Data Science wiki: how the cohort's data was produced)
