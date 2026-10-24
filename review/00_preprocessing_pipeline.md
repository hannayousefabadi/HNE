# HNE Preprocessing Pipeline — Review Notes

> Purpose: a "come back later and remember what this does" reference.
> Covers: what each module does, what order things run in, what gets QC'd
> (and what doesn't), and known gotchas worth remembering.

---

## 1. What this pipeline does, in one sentence

For each patient, it loads their Visium spatial transcriptomics data + matched H&E image, tiles the slide into ~1mm² regions, keeps only the tiles that are tumor-rich enough, computes 5 pathway signature scores per spot using ssGSEA rank-based enrichment, aggregates them to tile-level means, generates quantile-based binary calls, and saves the result — along with cropped H&E tile images — for downstream feature extraction and distributional regression modeling.

---

## 2. Where things live

```
src/hne/
├── core/
│   ├── config.py          # initial core output dirs
│   ├── paths.py           # PatientS3Paths, other output dirs, cohort manifest loading 
│   ├── s3_io.py           # S3DataLoader — all raw reads from S3 go through here
│   └── data_io.py         # thin wrappers: load_visium, load_spots, load_he_image, save_*
├── preprocessing/
│   ├── pipeline.py         # preprocess_patient() — the orchestrator, one call per patient
│   ├── preprocessing_config.py # gathered all the thresholds used in preprocessing
│   ├── tumor_purity.py     # tumor fraction, tile grid, Bayesian purity, tumor tile filter
│   ├── tiling.py            # crops + saves H&E tile PNGs
│   ├── spot_signatures.py  # compute_signatures() — per-spot pathway scores
│   └── aggregation.py      # spot→tile aggregation, ssGSEA aggregation + binary calls
└── preprocessing_qc/
    ├── tracker.py           # QCTracker — records OK/FLAG/EXCLUDE per stage, computes verdict
    └── plots.py             # per-patient + cohort QC plots

scripts/preprocessing/
└── preprocess_cohort.py    # loops preprocess_patient() over all 151 patients, aggregates outputs
```

**Rule of thumb:** `src/hne/` = reusable logic, importable. `scripts/` = the thing you actually run.

---

## 3. Per-patient pipeline flow (`preprocess_patient()` in `pipeline.py`)

Runs in this exact order — each stage's output feeds the next:

1. **Load raw data** (`data_io.py` → `s3_io.py`)
   - `vis` = AnnData (`.h5ad`) with counts, layers, deconvolution results
   - `spots` = tissue_positions.csv (spot coordinates)
   - `scalefactors_json.json` is **not** loaded. No scale factor is used anywhere in preprocessing: spot coordinates are already pixels of the H&E scan.
   - `slide` = `load_he_slide()` downloads the patient's full-resolution H&E scan (`{PROCESSED_VERSION}/converted_he/`) and opens it with OpenSlide as it is, with no conversion. It is opened after the tumor-fraction stage, so patients excluded there are never downloaded, and it stays open through tiling.
   - `fullres_pixel_size` = `slide_um_per_px(slide)`: µm per pixel read from the scan's own metadata (0.525 for the scans checked). **This is the only source of physical scale.** It raises if the scan carries no pixel size.

2. **`attach_tumor_fraction()`** *(tumor_purity.py)*
   - Pulls `vis.obsm["deconv_sample_level_custom1_frac"]`, sums columns containing `"Tu_"` → per-spot tumor fraction
   - Merges onto in-tissue spots
   - QC: `EXCLUDE` if no in-tissue spots, or mean tumor fraction < 0.2. `FLAG` if >10% of spots missing a tumor fraction value.
   - **Known gap:** not every patient has this deconvolution slot (only `cohort_level_2` was run for some). Handled as a graceful `EXCLUDE` + early skip now — previously crashed the whole patient with an unhandled `KeyError`.

3. **`add_tile_coordinates()`** *(tumor_purity.py)*
   - Converts `target_physical_size_um` (1000 µm = 1 mm) into a tile size in pixels of the scan: `round(1000 / fullres_pixel_size)`, 1,904 px at 0.525 µm/px.
   - **Scale check:** Visium spots are 100 µm apart centre to centre. The function measures the registered spot pitch in pixels, converts it with the scan's pixel size, and stores it as `spot_pitch_um_measured`. It logs a warning when that is more than 15% from 100 µm (`spot_pitch_tolerance`), which points to a registration or pixel-size problem for that patient. Across the cohort the expected value is about 99 µm, with three patients outside the tolerance.
   - **Rule: never derive pixel size from `spot_diameter_fullres`.** That value spans about 73 µm, not the 55 µm spot diameter, and gave tiles of about 1.4 mm.
   - Assigns every spot to a tile via floor division on pixel coordinates → `tile_id = "{tile_row}-{tile_col}"`
   - No QC record (just a logger warning if fewer than 30 initial tiles)

4. **`compute_tile_purity()`** *(tumor_purity.py)*
   - Bayesian shrinkage estimate per tile: `(sum(tumor_fraction) + 0.5*k) / (n_spots + k)`, prior mean 0.5, `k=2` by default
   - QC: `FLAG` if mean tile purity < 0.3

5. **`filter_tumor_tiles()`** *(tumor_purity.py)*
   - Keeps tiles with `tile_purity >= tumor_threshold` (default 0.3) **and** `n_spots >= min_spots` (default 40)
   - QC: `EXCLUDE` if 0 tumor tiles survive, `FLAG` if <10 survive
   - **This is the only QC stage that actually halts the pipeline early** — `pipeline.py` explicitly checks `has_tumor_tiles` and returns `(metadata, None, None, None)` if it's zero. Every other `EXCLUDE` elsewhere is currently *logged only* unless you've since added the same early-return pattern to it.

6. **`crop_and_save_tiles()`** *(tiling.py)*
   - Crops tiles directly via `OpenSlide.read_region()` at native resolution, clamps against slide dimensions to prevent boundary artifacts, pads edge crops to square, and saves PNGs to `TILES/{patient_id}/tile_r{row}_c{col}.png`.

   - ⚠️ **Naming mismatch to remember:** this saves files as `tile_r{row}_c{col}`, but the `tile_id` used everywhere else (signature CSVs, aggregation) is `"{row}-{col}"`. They refer to the same tile but aren't string-identical — don't assume you can `.png` lookup directly from a `tile_id` value without reformatting it.

7. **`compute_signatures()`** *(spot_signatures.py)*
   - Uses `vis.layers["log_norm_count"]` (log1p CPM) to retain zero-inflation structure.
   - Executes single-sample Gene Set Enrichment Analysis via `gseapy.ssgsea` across all 5 pathway signatures simultaneously.
   - Computes rank-based enrichment scores per spot, eliminating sequencing depth bias and preserving absolute cross-patient comparability without requiring destructive patient-level standardization.
   - QC: Logs detected vs. missing genes per signature; issues `EXCLUDE` if all 5 signatures fail, or `FLAG` if any signature is missing genes.
   - **Target Schema & Missing Signature Representation (`np.nan` vs `0.0`):**
   - All 5 pathway columns (`f"{sig}_score"`) are guaranteed to exist across every patient slide via uniform reindexing against the master pathway list.
   - When zero genes for a specific pathway are captured in a patient's slice (e.g., low-depth dropout in small signatures like YAP or EMT), the score is stored as `np.nan`, **never** `0.0`.
   - *Biological Rationale:* ssGSEA enrichment scores are zero-centered relative ranks; setting an unmeasured pathway to `0.0` falsely injects a signal of "average/neutral pathway activity." `np.nan` cleanly marks technical unobservability and preserves the true underlying target distribution.

8. **`aggregate_signatures()`** *(aggregation.py)*
   - Averages spot-level ssGSEA scores across all spots within each `tile_id` to produce continuous tile-level pathway scores (`f"{sig}_score"`).
   - Generates full-resolution bounding box coordinates (`x_min_fullres`, `y_min_fullres`, `x_max_fullres`, `y_max_fullres`).
   - Filters rows down to surviving qualified tumor tiles.
   - Issues a QC `FLAG` if fewer than `cfg.min_final_tumor_tiles` (10) survive.

9. **`binary_scores()`** *(aggregation.py)*
   - Operates directly on the raw, continuous tile-level ssGSEA enrichment scores.
   - Flags tiles in the top quartile (`quantile_threshold=0.75` by default) as positive (`1`, else `0`) under `f"{sig}_binary"`.
   - Leaves the continuous `f"{sig}_score"` values untouched for downstream distributional regression ($X, y$).
   - Saves output to `tile_signature_matrix/tiles_signature_matrix_{patient_id}.csv` and aggregates into `tiles_signature_matrix_cohort.csv`.
   - *Quantile Binarization Guard:* In `binary_scores()`, columns that are entirely `np.nan` remain `np.nan` in their respective binary target column (`f"{sig}_binary"`), preventing skewed thresholding or invalid comparisons during quantile calculations.

10. **QC plots** *(optional, `run_qc_plots=True`)* — signature variation, distribution, sparsity, consistency, correlation, per patient.

---

## 4. The QC verdict system

Three severities, computed per patient by `QCTracker.get_patient_verdict()`:

| Verdict | Meaning | Currently enforced? |
|---|---|---|
| `OK` | No FLAG or EXCLUDE at any stage | — |
| `REVIEW` | At least one `FLAG` somewhere | **Not automatically dropped** — deliberately kept in cohort outputs for now, meant for manual review later |
| `EXCLUDE` | At least one `EXCLUDE` somewhere | Only enforced if `preprocess_cohort.py` explicitly filters on verdict after the loop (added separately — check it's actually in place) |

QC stages that can fire (`tracker.py`'s `VALID_QC_STAGES`): `fullresimg_load`, `tumor_fraction`, `tile_purity`, `filter_tumor_tiles`, `signature_qc`.

**Why OK + REVIEW are both kept for now, and only EXCLUDE is dropped:**
`EXCLUDE` conditions are all "this patient's data is structurally unusable" —
zero tumor tiles, zero in-tissue spots, all 5 signatures failed to compute.
There's nothing to salvage; keeping them in the cohort would mean rows with
no real signal (or missing values that'd need to be dropped downstream anyway).

`FLAG` conditions (→ `REVIEW` verdict) are all "this patient's data is thinner
or noisier than ideal, but not necessarily wrong" — e.g. mean tile purity
just under 0.3, 10–something tumor tiles instead of comfortably more, >10%
of spots missing a tumor fraction value. These are soft warnings about
*confidence*, not evidence the data is bad. Dropping them automatically would
mean losing usable patients over a threshold that's somewhat arbitrary
(0.3 purity, 10 tiles) without ever looking at whether that specific patient
is actually fine. So the decision was: keep REVIEW patients in the cohort
output as usable-but-flagged, and manually look at the flagged list later
rather than have the pipeline silently decide for you.

**Note on `REVIEW` cases** Patients with flagged checks remain included in downstream feature extraction by default. When rigorous cohort filtering is required, filter on verdict == "OK" in qc_summary.csv before training.

**Important nuance:** a patient can crash with an unhandled Python exception (e.g. missing S3 file, missing obsm key not yet handled) *before* any QC record is written for them. Those patients show up in `preprocess_cohort.py`'s `failed_patients` list and the log traceback — **not** in `qc_summary.csv` at all. So "how many patients did I lose" = `EXCLUDE` verdicts in `qc_summary.csv` **plus** `failed_patients` from the cohort log — check both.

Outputs: `qc_reports/preprocessing_qc/{mode}/qc_records.csv` (one row per stage per patient) and `qc_summary.csv` (one row per patient, rolled up with a verdict).

---

## 5. Data sources (S3 layout, per patient)

> [!IMPORTANT]
> **Canonical Full-Resolution H&E Image Location:**
> - **Full-Resolution H&E Scans:** `s3://{PROCESSED_VISIUM_BUCKET}/{PROCESSED_VISIUM_PREFIX}/{PROCESSED_VERSION}/converted_he/{filename}`
> - **Filename mapping:** Loaded via `HE_MAP` in `src/hne/core/paths.py` from `results/cohort_metadata/cohort_manifest.json` (maps `{patient_id}` $\to$ `{patient_id}_vis.tif` for all 151 patients).
> - **Disallowed path:** `s3://{RAW_DATA_BUCKET}/.../Visium/image_files/` contains only low-resolution 3000 × 3000 px CytAssist preview images; it must never be read for preprocessing crops or feature extraction.

Built by `PatientS3Paths` in `paths.py`:

- **Processed Visium (`.h5ad`)**: `{PROCESSED_VISIUM_BUCKET}/{PREFIX}/{PROCESSED_VERSION}/without_spotclean/stLearn/{patient_id}_vis/{patient_id}_vis_c2l_annots.h5ad`
- **Spot positions / scale factors**: `.../{PROCESSED_VERSION}/spaceranger_count/{patient_id}_vis/outs/spatial/`
- **Full-resolution H&E**: `.../{PROCESSED_VERSION}/converted_he/{filename}`, where `{filename}` comes from `cohort_manifest.json`'s `he_map[patient_id]` (`{patient_id}_vis.tif`). This is the scan Space Ranger registered the spots to, so `pxl_*_in_fullres` are its pixels.
- **`PROCESSED_VERSION`** (`src/hne/core/config.py`, `"v2"`) is the one place the pipeline version is set. The h5ad, the spot positions and the H&E are all read from that version folder. **Rule: never read the image from one version and the coordinates from another.**
- **Not an H&E source: `{RAW_DATA_BUCKET}/.../Visium/image_files/`.** These are the 3000 × 3000 px CytAssist instrument images. The manifest keeps their names in `cytassist_map` for the registration audit; `load_cytassist_slide()` is the only function that opens them, and nothing is cropped from them.

`.h5ad` layers available (per the source pipeline's own docs): `raw_count`, `norm_count` (CPM), `log_norm_count` (log1p CPM — what this pipeline uses), `SCTransform`, `SME_normalized` (deprecated).

---

## 6. Known gotchas (worth remembering before debugging from scratch again)

- **`gseapy.ssgsea` returns long/tidy format matrices**: `res.res2d` produces a long table of $(N_{\text{spots}} \times N_{\text{signatures}})$ rows (e.g. $4{,}114 \times 5 = 20{,}570$ rows), **not** an aligned $(N_{\text{spots}}, 5)$ matrix. It must be dynamically pivoted on `Term` and `Sample`/`Name` before reindexing against `vis.obs_names`. Direct assignment of `vis.obs_names` to `res2d.index` will throw a length mismatch `ValueError`.
- **Dynamic signature drops**: Pathways with low gene counts (e.g., YAP or EMT with 6 genes) can drop to 0 detected genes in sparse slides. `compute_signatures()` dynamically handles dropped pathways and forces `.reindex(columns=all_signature_names, fill_value=0.0)` so all downstream matrices consistently contain all 5 target columns across the entire cohort (but some might be NAN of course).
- **S3 path double-prefixing**: If a bucket name constant is stored as a full `s3://...` URI instead of a bare name, `_parse_s3_path`'s single `.startswith("s3://")` strip only removes the outer one, leaving `Bucket="s3:"` → `NoSuchBucket`. Fixed via `urlsplit`-based parsing.
- **`HE_MAP` and `CYTASSIST_MAP` entries must be bare filenames**, not full paths — same double-prefix risk if the manifest is edited by hand.
- **Tiles the scan does not cover are dropped.** The Visium capture area can reach past the edge of the H&E scan, so a few spots have no image under them. `drop_tiles_outside_slide()` keeps only tumor tiles that lie fully inside the scan and records `n_tiles_outside_slide`. If more than half of a patient's tumor tiles are outside, it raises: that means the wrong image was opened. `crop_and_save_tiles()` still raises on a tile that starts outside the slide, as a backstop.
- **Cohort inventory** (`scripts/cohort_inventory/cohort_inventory.py`) builds the manifest from `{PROCESSED_VERSION}/spaceranger_count/` and `{PROCESSED_VERSION}/converted_he/`. A patient is `MATCHED` only with both, and only if the H&E file is at least 100 MB (`HE_TOO_SMALL` otherwise: a scan is hundreds of MB, a CytAssist image under 30). Other statuses: `MISSING_HE`, `HE_HAS_NO_VISIUM_MATCH`.
- **`tile_id` string format differs between `tiling.py` (`tile_r{row}_c{col}`) and everywhere else (`"{row}-{col}"`)** — see step 6 above.
- **Missing `deconv_sample_level_custom1_frac`**: Not guaranteed present for every patient — only patients with a patient-specific deconvolution signature matrix have it. Handled gracefully (`EXCLUDE` + skip), not an unhandled exception.
- **Feature extraction resume logic**: `run_phikon_extractor.py` skips patients with no `tiles_signature_matrix_{patient_id}.csv` (never preprocessed) and skips patients whose `.npy` features already exist (already extracted) — a *patient-level* resume check, not tile-level.

---

## 6b. Memory: what is held, and for how long

Preprocessing runs one patient at a time. What each stage keeps in RAM:

| Object | Size | Handling |
|---|---|---|
| H&E scan (0.05 to 1.9 GB file) | About 1 MB to open | Downloaded to a temp file on disk and opened lazily by OpenSlide; only the regions read are decoded. The temp file is deleted when the patient's slide context closes |
| Tile images | About 11 MB per 1,904 px tile, one at a time | `crop_and_save_tiles()` writes each tile and releases it before reading the next. Measured: +64 MB for 30 tiles. It must not collect the images in a list: that held 0.4 to 0.9 GB per patient through the rest of the pipeline |
| AnnData (`.h5ad`, 0.7 to 1.6 GB file) | About 130 MB after loading | `load_visium(keep_layers=["log_norm_count"])` drops every other layer. The dense `SCTransform` layer alone is about 0.6 GB and nothing in the pipeline reads it |
| ssGSEA input | Dense spots × genes matrix, about 0.3 GB as float32, copied into each worker | **The largest remaining consumer, and not measured.** `ssgsea_processes` (4) multiplies it. Lower that value first if memory runs out |
| Per-patient results kept for the cohort | Spot and tile tables, a few MB each | Accumulated in `preprocess_cohort.py` for the cohort CSVs and plots |

- `preprocess_cohort.py` calls `gc.collect()` after every patient, because AnnData objects hold reference cycles and are otherwise freed late.
- **Temp directory.** Slides are downloaded to Python's temp directory (`TMPDIR`, default `/tmp`). If that is a RAM-backed filesystem (tmpfs), a 1.9 GB download counts as memory. `open_slide()` logs a warning in that case; point `TMPDIR` at a directory on disk. The disk needs room for the largest scan.

---

## 7. Cohort-level orchestration (`preprocess_cohort.py`)

- Loads `PATIENT_IDS` from `cohort_manifest.json`
- Loops every patient through `preprocess_patient(mode='cohort', ...)` inside a `try/except` — failures are logged and added to `failed_patients`, loop continues (doesn't crash the whole run)
- Collects `all_metadata`, `all_tiles_sig`, `all_spot_data` across patients
- Saves: cohort `metadata.csv`, `tiles_signature_matrix_cohort.csv`, cohort QC plots
- Prints final summary: total discovered / successfully processed / failed, plus the QC verdict counts

**To answer "what did I keep vs. drop" for any given run, check three things together:**
1. `qc_summary.csv["verdict"].value_counts()` — OK / REVIEW / EXCLUDE among patients that got QC records
2. The cohort log's final `Failed patients: [...]` list — patients that crashed before any QC record was written
3. Whether the EXCLUDE-enforcement filter is actually present in `preprocess_cohort.py` (drops `EXCLUDE`-verdict patients from `all_tiles_sig`/`all_metadata`/`all_spot_data` before saving) — without it, `EXCLUDE` patients' data may still be sitting in your final cohort CSV despite the verdict.

---

## 8. Configurable Thresholds & Central Configuration

All pipeline thresholds, geometric settings, and QC tolerances are managed through a single immutable configuration instance (`PREPROCESSING_CONFIG`) defined in `src/hne/preprocessing/preprocessing_config.py`. Modules and scripts import this instance directly, ensuring parameter changes apply globally.

| Configuration Field | Default Value | Role & Operational Boundary |
|---|---|---|
| `target_physical_size_um` | `1000.0` (1 mm) | Physical tile edge length used to calculate tile size in native pixels (`round(target / fullres_pixel_size)`) |
| `spot_diameter_um` | `55.0` (µm) | Physical diameter of a Visium spot. Not used to compute anything in preprocessing; the registration audit uses it to reproduce the old pixel size |
| `spot_pitch_um` | `100.0` (µm) | Centre-to-centre spot distance, the reference for the scale check |
| `spot_pitch_tolerance` | `0.15` | Relative deviation of the measured pitch that triggers a warning |
| `min_initial_tiles` | `30` | Minimum initial grid tiles before issuing a layout warning |
| `k_prior` | `2.0` | Bayesian shrinkage pseudo-count weight: `(alpha + 0.5*k) / (n_spots + k)` |
| `mean_tumor_fraction_threshold` | `0.2` (20%) | Minimum patient-level mean spot tumor fraction (`EXCLUDE` if below) |
| `missing_tumor_fraction_tolerance` | `0.1` (10%) | Spot missingness tolerance (`FLAG` if missing / total in-tissue spots > 10%) |
| `mean_purity_threshold` | `0.3` (30%) | Minimum patient-level average Bayesian tile purity (`FLAG` if below) |
| `tumor_purity_threshold` | `0.3` (30%) | Minimum tile Bayesian purity required to retain a tile |
| `min_spots_per_tile` | `40` | Minimum spot count required to retain a tile. A full 1 mm tile holds about 90 to 100 spots, so this is roughly 40% coverage (it was about 20% when tiles were 1.4 mm) |
| `min_final_tumor_tiles` | `10` | Minimum surviving tumor tiles before issuing a QC `FLAG` |
| `ssgsea_processes` | `4` | Number of worker processes allocated for `gp.ssgsea` spot scoring |
| `ssgsea_min_size` | `1` | Minimum genes required to score a pathway signature |
| `binary_quantile_threshold` | `0.75` (p75) | Quantile cutoff for binary classification (`1` if score $\ge$ 75th percentile) |
| `sparse_threshold` | `1e-6` | Value below which spot scores are treated as zero in sparsity QC plots |
| `min_tissue_fraction` | `0.5` (50%) | Minimum non-glass tissue area (`gray < 220`) required to accept an FM sub-patch |

---

## 9. Appendix: Historical Image Registration Audit Summary

*(Consolidated from legacy audit documents `05_registration_audit_note.md` and `06_registration_audit.md`)*

During initial baseline development (October 2026), the linear baseline initially yielded zero correlation and failed the tumor purity positive control. A comprehensive cohort audit identified the root cause:

1. **Root Cause:** The cohort inventory manifest initially pointed to `spatial_transcriptomics/Visium/image_files/`, which contained low-resolution CytAssist preview images (`~3000 × 3000` px) rather than the actual microscope scans. Space Ranger spot coordinates (which reach up to ~34,000 px) fell outside the image, causing edge-clamping, black crops, and duplicate feature vectors.
2. **Permanent Fixes:**
   - Switched slide loading strictly to the full-resolution scans at `s3://.../{PROCESSED_VERSION}/converted_he/{patient_id}_vis.tif` (mapped via `HE_MAP` in `src/hne/core/paths.py`).
   - Read native slide pixel size directly from slide metadata (`~0.525 µm/px`) rather than estimating from spot diameter.
   - Removed all silent coordinate clamping; out-of-bounds patches now fail loudly or are explicitly counted in QC logs.
3. **Outcome:** Rerunning Phase 1 Ridge on the corrected full-resolution H&E scans completely restored the biological signal across the cohort ($r_{\text{within}} \approx 0.29\text{--}0.39, p = 10^{-4}$) and the purity positive control ($r = 0.79$).


