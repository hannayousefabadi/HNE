# HNE: predicting tumor pathway activity from H&E histology

A pipeline for lung cancer that learns to read pathway activity from routine H&E slides. Matched Visium spatial transcriptomics provides the ground truth: each slide is cut into ~1 mm² tumor tiles, each tile is scored for 5 pathway signatures (FMRP, Cell cycle, YAP, WNT, EMT), and a model is trained to predict those scores from the tile image alone.

## How it works

1. **Cohort inventory.** Lists the patients that have both Space Ranger output and a full-resolution H&E scan on S3, and writes a manifest.
2. **Preprocessing.** Per patient: estimates tumor purity, tiles the slide, keeps tumor-rich tiles, scores each Visium spot with ssGSEA, and averages spot scores to tile level. Every patient gets a QC verdict.
3. **Feature extraction.** Embeds each tile with the Phikon-v2 pathology foundation model.
4. **Phase 1 baseline.** Fits a ridge regression from tile embeddings to signature scores and evaluates it with nested cross-validation grouped by patient.

## Install

Developed on Python 3.10.

```bash
pip install -r requirements.txt
```

This installs the dependencies and the `hne` package in editable mode.

## Configure

Raw data is read from S3. Create a `.env` file in the repository root:

```
PROCESSED_VISIUM_BUCKET=...
PROCESSED_VISIUM_PREFIX=...
RAW_DATA_BUCKET=...
RAW_DATA_PREFIX=...
```

AWS credentials are picked up the usual way (environment variables, `~/.aws`, or an instance role).

## Run

Run the stages in order. Each one reads what the previous one wrote. Settings are edited at the top of each script; there are no command-line flags.

```bash
# 1. Build the cohort manifest (required before anything else)
python scripts/cohort_inventory/cohort_inventory.py

# 2. Preprocess every patient in the manifest
python scripts/preprocessing/preprocess_cohort.py

# 3. Extract Phikon-v2 features (GPU recommended; rerun to resume; progress in qc_reports/feature_extraction_qc/)
python scripts/feature_extraction/run_phikon_extractor.py

# 4. Fit and evaluate the Phase 1 ridge baseline
python scripts/model_train/phase1/train_phase1_ridge.py
```

To try preprocessing on one patient first, set the patient ID inside `scripts/preprocessing/preprocess_single_patient.py` and run it.

## Outputs

| Path | Content |
|---|---|
| `results/cohort_metadata/` | Cohort manifest and inventory reports |
| `tiles/` | Cropped H&E tile images, one folder per patient |
| `tile_signature_matrix/` | Tile-level signature scores, one CSV per patient plus a cohort CSV |
| `qc_reports/preprocessing_qc/` | QC records, per-patient verdicts, logs, and QC plots |
| `feature_sets/phikon_v2_features/` | One 1024-d embedding per tile (`.npy`) |
| `results/phase1_ridge/` | Phase 1 baseline: cross-validated predictions, metrics, and run summary |
| `results/phase1/` | Earlier MLP experiment, kept for comparison |
| `results/registration_audit/` | Image-input audit: slide inventory, overlays, sample patches |
| `qc_reports/feature_extraction_qc/` | Feature extraction logs per patient and per tile |
| `plots/` | Figure scripts and figures |

## Project structure

```text
HNE_repo/
├── scripts/                 # Entry points, one folder per stage
│   ├── cohort_inventory/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── model_train/phase1/
│   └── audit/               # Image-input and registration audit
├── src/hne/                 # Importable package
│   ├── core/                # Paths, configuration, S3 and data I/O
│   ├── preprocessing/       # Purity, tiling, signatures, aggregation
│   ├── preprocessing_qc/    # QC tracker and QC plots
│   ├── feature_extraction/  # Patching and Phikon-v2 extractor
│   └── models/              # Data loading, evaluation, and the MLP
├── plots/                   # Figure scripts
├── tests/                   # Debugging and diagnostic scripts
├── requirements.txt
└── pyproject.toml
```

## Status

The cohort has 151 patients, of which 93 pass preprocessing QC. The pipeline reads the full-resolution H&E scans and cuts true 1 mm tiles. Preprocessing has been rerun on that basis (91 patients, 2,989 tumor tiles; 2 patients to redo). Feature extraction is partly done (58 patients). The Phase 1 results in `results/` predate these fixes and are not meaningful; they are regenerated once extraction is complete.
