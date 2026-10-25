# Roadmap and Implementation Plan:

Collected during planning discussions (Oct 2026). The final step-by-step plan must include everything here. Terminology: `review/00_glossary.md`.

## Project premise

- Goal: link H&E tiles (1 mm, TMA scale) to intratumoral heterogeneity of pathway signatures, using tiles to amplify sample size beyond the number of patients.
- Tiles are used as **training samples** (legitimate). Generalization to new patients is still measured in **patients** (statistics at patient level).
- The premise "tiles carry independent information" is tested directly: report ICC per signature and the effective sample size. Low ICC = most variance is within patients = strong support for tile-level amplification.

## Targets and outputs

- Targets: tile signature score vector, all 5 signatures (FMRP, Cell cycle, YAP, WNT, EMT), **cohort z-scored** with training-fold statistics. (cohort z-scoring does not ruin within-patient differences. The raw scores for the 5 signatures (FMRP, Cell cycle, YAP, WNT, EMT) have different numerical ranges and variances. Without z-scoring, the loss function (MSE or β\beta**β**-NLL) would place 90% of its gradient weight on whichever signature happens to have the largest numerical scale, starving the others.)

  * Targets are z-scored using training-fold statistics, and after inference, predictions (μ^,σ^) are back-transformed to raw biological score units before computing all final metrics.
  * **Should you drop it? No.**Cohort z-scoring is purely a numerical tool to help the optimizer converge fairly across all five pathways. Because all outputs are back-transformed to raw units before reporting, you get the numerical stability during training with zero loss of raw biological interpretability.
- Distributional regression: predict μ and σ. Train σ with Gaussian NLL (try a stabilized variant: μ warm-up first, and/or β-NLL). Validate σ: calibration (coverage of μ ± σ) and error-vs-σ correlation. Optional: compare predicted σ with the measured spread of spot scores within each tile.

## Feature sets

- Single models: Phikon-v2 (current), Virchow2, CONCH v1.5. Each extracted with its own patch spec (already in `patching.py`).
- Fusion arms: naive concatenation, intelligent fusion (correlation-guided pruning, Flannery et al. arXiv:2512.11104, adapted to regression), AdaFusion (Xiao, Hu et al. arXiv:2608.01370, code: github.com/xyx-98/PathoOracle).
- Intelligent fusion is the feature selection step; no separate open-ended feature selection.

The project now faces a **combinatorial explosion**:

- **3 Foundation Models (FMs):** Phikon-v2 (1,024-d), Virchow2 (2,560-d), CONCH v1.5 (768-d).
- **3 Fusion / Selection Schemes:** Naive Concatenation, Intelligent Fusion (correlation-guided pruning), AdaFusion (learned gating).
- **3 Aggregation / Architectures:** Mean-pooled Ridge, Mean-pooled MLP, Attention-Based Multiple Instance Learning (ABMIL over patch bags).
- **2 Regression Formulations:** Scalar point regression ($\mu$) vs. Distributional regression ($\mu, \sigma$ trained with Gaussian NLL).

A naive grid search ($3 \times 3 \times 3 \times 2 = 54$ models $\times 25$ folds = 1,350 fits) would be computationally intractable, statistically unsound, and impossible to publish cleanly.

This implementation plan resolves the combinatorial explosion by **factorizing the search into a sequence of controlled, scientific phases**. Each phase isolates exactly **one axis of variation** while holding all other variables constant, benchmarking against the frozen Phase 1 numbers under the identical nested grouped CV harness.

## Phased Progression Matrix

To avoid combinatorial explosion, the table below defines which component is varied at each step while all others are held constant:

| Phase                          | Primary Research Question                                                        | Inputs                                       | Head Architecture                 | Objective Function                                          | Baseline to Beat                                                | Key Results                                                                                                                       | Decision / Methodological Impact                                                                                                                    |
| ------------------------------ | -------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** *(Done)*   | Linear baseline on single FM                                                     | Phikon-v2 (Mean)                             | Ridge Regression                  | MSE ($L_2$)                                               | N/A (Anchor Baseline)                                           | Within-patient$r = 0.29\text{--}0.39$ ($p=10^{-4}$ across all 5 signatures); ICC $= 0.63\text{--}0.76$                      | **Validates core premise:** H&E captures intratumoral heterogeneity. Frozen as permanent anchor baseline.                                     |
| **Phase 1b** *(Done)*  | Does non-linearity help? Can we train calibrated $\sigma$?                      | Phikon-v2 (Mean)                             | Distributional MLP                | Gaussian $\beta$-NLL ($\beta=0.5$)                          | Phase 1 Ridge                                                   | Distr MLP $r_{\text{within}} = 0.27\text{--}0.34$ (did not beat Ridge); $\beta$-NLL calibrated $\sigma$ for Cell_cycle & EMT | **Non-linear heads alone do not improve ranking:** Proves mean-pooling is the true bottleneck. **Mandates Phase 2 (ABMIL patch bags).** |
| **Phase 2** *(Active)* | Does learned patch attention beat equal-weight mean pooling?                     | Phikon-v2 (Unpooled Patch Embeddings)        | ABMIL (Attention Pooling + Head)  | Best from Phase 1b (Point or Distr)                         | Phase 1 Ridge ($r_{\text{within}} \approx 0.29\text{--}0.39$) | *Pending patch extraction & training*                                                                                           | Determines if learned spatial weighting captures localized hotspots and outputs meaningful attention entropy.                                       |
| **Phase 3**              | How do individual clinical FMs rank on their own?                                | 1. Virchow22. CONCH v1.5                     | Winning Head from Phase 1b/2      | Winning Objective                                           | Phikon-v2 ABMIL                                                 | *Pending extraction*                                                                                                            | Ranks clinical foundation models individually to select the top single-model baseline before multimodal fusion.                                     |
| **Phase 4a**             | Does naive concatenation beat the single best FM?                                | Naive Concat ($4,352$-d)                   | Winning Head                      | Winning Objective                                           | Best Single FM (Phase 3)                                        | *Pending*                                                                                                                       | Tests whether brute-force feature stacking outperforms the single best FM without feature selection.                                                |
| **Phase 4b**             | Does correlation-guided pruning (Intelligent Fusion) improve over concatenation? | Pruned Concat ($\theta$ tuned in inner CV) | Winning Head                      | Winning Objective                                           | Naive Concat & Best Single FM                                   | *Pending*                                                                                                                       | Determines whether removing cross-FM multi-collinearity and noise boosts generalization.                                                            |
| **Phase 4c**             | Does learned adaptive gating (AdaFusion) beat pruning and concatenation?         | Multi-FM Bags / Embeddings                   | AdaFusion Gating Network          | Winning Objective                                           | Intelligent Fusion & Naive Concat                               | *Pending*                                                                                                                       | Evaluates if dynamic learned gating (AdaFusion) delivers optimal multimodal synergy.                                                                |
| **Phase 5**              | Heterogeneity attribution & validation                                           | Winning Architecture                         | Winning Architecture              | Winning Objective                                           | Partial$r$ (purity-controlled), Intratile variance            | *Pending*                                                                                                                       | Proves biological validity beyond tissue purity confounder; links patch attention to Visium spot variance.                                          |

---

## ICC

ICC stands for Intraclass Correlation Coefficient: it measures how much of the total variance in a pathway score comes from differences between patients versus differences within the same tumor.

![1792769014863](image/00_roadmap_and_implementation_plans/1792769014863.png)

* If ICC=1.0: Every tile inside a tumor has the exact same score. There is zero intratumoral heterogeneity; all variation is between different patients.
* If ICC=0.0: Knowing which patient a tile came from tells you nothing. All variation is strictly local/within-tumor.

## Evaluation rules (apply to every phase)

1. Any data that influenced a decision cannot be used to report that decision's result. Decisions include: epoch/early stopping, hyperparameters (lr, hidden size, dropout, α), which FM / fusion wins, intelligent-fusion θ, AdaFusion d, which features survive, which signatures are shown.
2. All decisions are made inside grouped (by patient) CV on training patients; nested CV when tuning. The test set is used once, after everything is frozen.
3. Count patients, not tiles: CIs from patient-level bootstrap; p-values from patient-level permutation or mixed-effects models.
4. Report pooled, within-patient and between-patient r for every signature. The heterogeneity claim rests on within-patient r.
5. Report all 5 signatures, including weak ones (names may be masked in public figures).
6. Always run the ridge baseline next to anything more complex.
7. Scripts fail loudly; no hardcoded fallback numbers.
8. Do not claim an output (e.g. σ) until it is trained and validated.
9. Write down the metrics, signatures and comparisons to report before looking at test results.

## Project Decisions Log

The table below records all major methodological and architectural decisions, the rationale behind them, the empirical or theoretical evidence they are based on, and their current status:

| #            | Topic / Area                                       | Decision Taken                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Rationale / Why We Made This Choice                                                                                                                                                                                                       | Based on What Evidence / Results?                                                                                          | Originating Phase                | Status            |
| ------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------- |
| **1**  | **Image Source & Scale**                     | Extract exclusively from full-resolution microscope H&E (`converted_he/{patient}_vis.tif` at ~0.525 µm/px). CytAssist images are strictly forbidden.                                                                                                                                                                                                                                                                                                             | Space Ranger spot coordinates reach up to ~34,000 px and only align with full-res scans. CytAssist images (~3000 px) caused out-of-bounds crops, coordinate clamping, and zero signal.                                                    | Registration Audit (`06_registration_audit.md`, commit `c30f502`). Spot-aligned purity control restored to $r=0.79$. | Preprocessing Audit (2026-10-08) | **Locked**  |
| **2**  | **Tile Definition & Bag Semantics**          | Tiles are 1 mm × 1 mm tissue squares (TMA scale, ~1,904 px) with$\ge 40$ spots and tumor purity $\ge 0.3$. "Bag" strictly means **Tile Bag** (all ~80 sub-patches belonging to one 1 mm tile).                                                                                                                                                                                                                                                           | Matches standard clinical tissue microarray punch cores; guarantees sufficient transcriptomic coverage and viable tumor content. Patient-level bags (92 samples) were rejected.                                                           | Glossary (`00_glossary.md`) & Preprocessing pipeline.                                                                    | Preprocessing (2026-10-09)       | **Locked**  |
| **3**  | **Cohort Z-Score Normalization**             | Target scores are standardized using mean and std computed**strictly on training-fold patients**, back-transforming test predictions to raw units. Per-patient z-scoring is strictly forbidden.                                                                                                                                                                                                                                                               | Per-patient z-scoring destroys between-patient differences (hot vs cold tumors) and distorts within-patient heterogeneity amplitude. Cohort z-scoring linearly preserves all within-tumor contrasts while balancing multi-task gradients. | Mathematical linearity proofs & Phase 1 CV harness.                                                                        | Phase 1 (2026-10-10)             | **Locked**  |
| **4**  | **CV Scheme & Statistical Unit**             | 5-fold outer$\times$ 5-fold inner nested cross-validation grouped by patient (`GroupKFold`). Patient-level bootstrap (2,000 resamples) and patient-level permutations (10,000). Tile-level $p$-values are forbidden.                                                                                                                                                                                                                                          | Visium signature scores have true ICC$= 0.63\text{--}0.76$, meaning 2,993 tiles carry the power of only $\approx 120\text{--}145$ independent observations. Grouping by patient prevents pseudoreplication and data leakage.          | ICC calculations in`evaluation.py` & Rules 2, 3.                                                                         | Phase 1 (2026-10-11)             | **Locked**  |
| **5**  | **Phase 1 Anchor Baseline**                  | Plain linear Ridge regression on mean-pooled Phikon-v2 tile embeddings ($r_{\text{within}} \approx 0.29\text{--}0.39, p=10^{-4}$).                                                                                                                                                                                                                                                                                                                                | Closed-form, deterministic baseline with analytical L2 regularization. Establishes the minimum performance bar that any complex deep learning architecture must beat.                                                                     | Rerun on full-resolution H&E (`02_phase1_ridge_baseline.md`).                                                            | Phase 1 (2026-10-11)             | **Locked**  |
| **6**  | **Uncertainty Loss Formulation**             | Gaussian$\beta$-NLL ($\beta=0.5$) with detached variance weighting over standard Gaussian NLL or Pearson-Huber loss. | Pearson-Huber cannot train uncertainty ($\sigma$). Standard Gaussian NLL suffers from uncertainty explosion on hard outliers. $\beta=0.5$ downweights outlier gradients, preventing $\sigma \to \infty$ collapse. | Seitzer et al. (ICLR 2022); calibrated coverage (60.7%) and error correlation ($\rho = +0.252$) in Phase 1b. | Phase 1b (2026-10-21)                                                                                                                                                                                                                     | **Locked**                                                                                                           |                                  |                   |
| **7**  | **Neural Head Architecture**                 | Distributional architecture (`DistributionalMLP`: shared trunk branching into `mu_head` and `log_std_head`) as the sole neural head tested in Phase 1b (Point MLP was not coded or run). | Naturally outputs conditional expectation $\mu$ while providing calibrated uncertainty $\sigma$. $\beta$-NLL acts as an adaptive outlier-robust loss on $\mu$. | Phase 1b Distributional MLP (`src/hne/models/mlp.py`).                                                                   | Phase 1b (2026-10-21)            | **Locked**  |
| **8**  | **Rejection of Mean-Pooling for Deep Heads** | Neural MLP heads on mean-pooled vectors failed to outperform Ridge on within-patient ranking ($r_{\text{within}} \approx 0.27\text{--}0.34$ vs Ridge $0.29\text{--}0.39$).                                                                                                                                                                                                                                                                                      | Uniform mean-pooling over 80 patches washes out subtle spatial contrast between tumor, stroma, and necrosis. Deep heads memorize patient-level features rather than learning intra-tumor gradients.                                       | Phase 1b side-by-side benchmark (`03_phase1b_distributional_mlp.md`).                                                    | Phase 1b (2026-10-21)            | **Decided** |
| **9**  | **Phase 2 Mandate: ABMIL on Tile Bags**      | Transition from mean-pooled tile vectors to unpooled patch embeddings ($N_{\text{patches}} \times 1024$) pooled via Attention-Based Multiple Instance Learning (ABMIL over tile bags).                                                                                                                                                                                                                                                                          | Solves the mean-pooling bottleneck. Learns attention weights across sub-patches to detect localized pathway hotspots, while attention entropy provides an intrinsic morphological heterogeneity readout.                                  | Phase 1b bottleneck analysis & literature (Ilse et al.).                                                                   | Phase 2 Planning (2026-10-22)    | **Active**  |
| **10** | **Modular PyTorch Nested CV Harness**        | Create a standalone modular library module`src/hne/models/torch_cv.py` and verify parity against Phase 1b results before proceeding to Phase 2 training.                                                                                                                                                                                                                                                                                                          | Encapsulates patient-grouped outer CV, inner early stopping, leak-free scalers, and Phase 1-compliant output generation, avoiding code duplication across Phases 2, 3, and 4.                                                             | Architectural refactoring requirement.                                                                                     | Current (2026-10-23)             | **Active**  |
| **11** | **Tumor Purity Confounder Handling**         | Tumor purity correlates with pathway activity ($r \approx 0.3\text{--}0.5$) and explains roughly 25–33% of within-patient $r$.                                                                                                                                                                                                                                                                                                                                 | We must prove our models learn genuine pathway biology beyond simple tumor cell density.                                                                                                                                                  | Phase 1 partial correlation audit (mean partial$r = 0.24\text{--}0.30$ after removing purity).                           | Phase 1 / Phase 5                | **Open**    |

---

## Active Implementation Plan: Phase 2 (ABMIL on Tile Bags)

Phase 2 replaces uniform mean-pooling with **Attention-Based Multiple Instance Learning (ABMIL)** on tile bags.

### Component 1: Modular PyTorch Grouped Nested CV Harness (`src/hne/models/torch_cv.py`)

Reusable, leak-free engine implemented in `src/hne/models/torch_cv.py`:

- Outer `GroupKFold` by patient (5 folds, seed 42).
- Inner `GroupKFold` by patient (5 folds) for patient-level early stopping.
- Leak-free z-scoring on training folds only (`fit_scalers`).
- Standardized output generation matching Phase 1 schema (`cv_metrics_summary.csv`, `cv_predictions.csv`, `cv_per_patient_metrics.csv`, `cv_fold_assignments.csv`).
- Modular support for both tabular 2D arrays and variable-instance tile bags (`TileDataset`, `collate_tile_bags`).
- **Parity Verification Step:** Run Phase 1b Distributional MLP through `torch_cv.py` to confirm it reproduces Phase 1b metrics and calibration outputs before using it for Phase 2.

### Component 2: Regression Architecture & Loss Selection

- **Standard Head:** We proceed with the **Distributional Head** (predicting both $\mu$ and $\sigma$).
- **Relationship between Point $\mu$ and Distributional $\mu$:**
  - A Point MLP optimizes pure MSE: every tile exerts gradient force proportional to $(y - \mu)$.
  - Distributional MLP optimizes $\beta$-NLL ($\beta=0.5$): the gradient on $\mu$ is weighted by $\frac{1}{\sigma}$ (inverse predicted standard deviation). Confident tiles drive the weights while uncertain outliers are softly downweighted.
  - Using the Distributional Head gives both the outlier-robust $\mu$ prediction and the calibrated spatial uncertainty $\sigma$ within a single unified model.

### Component 3: Patch-Bag Feature Extraction Pipeline

- Phikon-v2 extractor updated with `--unpooled` flag (`scripts/feature_extraction/run_phikon_extractor.py`), storing unpooled patch embeddings $(N_{\text{patches}} \times 1024)$ in:
  `feature_sets/phikon_v2_patch_features/{patient_id}_{tile_id}_phikon_patch_features.npy`
- Registered in `src/hne/models/data.py` (`phikon_v2_unpooled`) and path defined in `src/hne/core/paths.py`.
- Patch extraction is currently running on SageMaker GPU (`ml.g4dn.xlarge`).

### Component 4: ABMIL Model Architecture (`src/hne/models/abmil.py`)

- **Gated Attention Mechanism (Ilse et al.):**
  $$
  a_m = \frac{\exp\left(w^\top \left[\tanh(V h_m) \odot \text{sigmoid}(U h_m)\right]\right)}{\sum_{j=1}^M \exp\left(w^\top \left[\tanh(V h_j) \odot \text{sigmoid}(U h_j)\right]\right)}
  $$
- **Aggregated Representation:** $z_{\text{tile}} = \sum_{m=1}^M a_m h_m$.
- **Heads:** Shared trunk projecting $z_{\text{tile}}$ to $\mu$ and $\log \sigma$ for all 5 signatures.
- **Attention Entropy:** Compute tile morphological diversity $H_{\text{tile}} = -\sum_{m=1}^M a_m \log(a_m + \epsilon)$ to test as an intrinsic image readout of intratumoral heterogeneity.

### Component 5: Phase 2 Training & Evaluation Script

- `scripts/model_train/phase2/train_phase2_abmil.py` using `src/hne/models/torch_cv.py`.
- Benchmark directly against Phase 1 Ridge ($r_{\text{within}} \approx 0.29\text{--}0.39$) and Phase 1b Distributional MLP ($r_{\text{within}} \approx 0.27\text{--}0.34$).

---

## Active Open Questions

1. **Tumor Purity in Evaluation (Phase 5):**
   - Option A: Report purity-adjusted partial Pearson $r$ alongside raw $r$.
   - Option B: Restrict evaluation to high-purity tiles (e.g. tumor purity $\ge 0.5$).
   - Option C: Fit a linear baseline model predicting signature scores directly from purity and measure image model gain over purity alone.
2. **Multi-Seed Stability:**
   - Single seed (42) is locked during model development for exact paired comparisons across all phases.
   - At project completion, a 3-seed or 5-seed repetition will be run as a final stability check.
