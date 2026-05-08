# H&E MIL - Visium Spatial Transcriptomics + H&E Image Analysis

Pipeline for preprocessing Visium spatial transcriptomics data with matched H&E images for Multiple Instance Learning (MIL).

## Overview

This pipeline processes paired Visium ST and H&E data to generate tile-level feature matrices ready for MIL model training.

**Key steps:**
- Tumor purity estimation (Bayesian shrinkage)
- H&E image tiling
- Pathway signature computation per spot
- Aggregation to tile-level features
- Quality control and cohort summarization

## Project Structure
```text
HNE_repo/
├── data/ # Patient data (Visium ST, spaceranger output)
├── scripts/
│ └── preprocessing/ # Run scripts (single_patient, cohort)
├── src/hne/
│ ├── core/ # Paths and I/O
│ ├── preprocessing/ # Tiling, purity, signatures, aggregation
│ └── preprocessing_qc/ # QCTracker and QC plots
├── tile_features/ # Output: tile signature matrices (CSV)
├── tiles/ # Output: cropped H&E tiles (PNG)
├── qc_reports/ # Output: QC summaries, logs, plots
└── pyproject.toml
```


## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2.  Configure patient paths
Update PATIENT_IDS in src/hne/core/paths.py with your patient IDs.

### 3. Run preprocessing
Single patient:

```bash
python scripts/preprocessing/preprocess_single_patient.py
```

Cohort:

```bash
python scripts/preprocessing/preprocess_cohort.py
```

