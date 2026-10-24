# Phase 2: Attention-Based Multiple Instance Learning (ABMIL) on Tile Bags — Review Notes

> Purpose: Comprehensive reference for Phase 2 architecture, unpooled patch representation, attention pooling, spatial interpretability, and intratumoral heterogeneity readouts.
> Covers: Architecture, gated attention formulation, attention weights as spatial signals, attention entropy ($H_{\text{tile}}$), distribution shape diagnostics, and the nested grouped CV evaluation protocol.
> Branch: `master`
> Terminology: `review/00_glossary.md`. Master plan & Decisions Log: `review/00_roadmap_and_implementation_plans.md`.

---

## 1. What it is

Phase 2 replaces uniform mean-pooling with **Attention-Based Multiple Instance Learning (ABMIL)** on tile bags. 

Instead of averaging all sub-patches into a single 1024-d vector before modeling (which erased the fine spatial contrast between tumor parenchyma, stroma, and necrosis in Phase 1 and 1b), Phase 2 operates on **unpooled patch embeddings**:
* **Input per tile:** A bag of patch embeddings $H_i \in \mathbb{R}^{M_i \times 1024}$, where $M_i$ is the number of valid tissue patches in that 1 mm tile (typically $M_i \approx 50\text{--}80$ sub-patches of 112 µm).
* **Attention Pooling:** A trainable gated neural network learns a scalar attention weight $a_m \in (0, 1)$ for every patch, aggregating the variable-sized bag into a single, highly informative tile embedding $z_{\text{tile}} \in \mathbb{R}^{1024}$.
* **Distributional Head:** $z_{\text{tile}}$ feeds into our standard dual head to predict the conditional expectation ($\mu$) and spatial uncertainty ($\sigma$) for all 5 pathway signatures, trained with Gaussian $\beta$-NLL ($\beta=0.5$).
* **Evaluation:** Evaluated with the exact same **nested grouped cross-validation harness (`src/hne/models/torch_cv.py`)** over the verified 92-patient cohort (2,993 tiles).

---

## 2. Where Things Live

```
src/hne/
├── feature_extraction/
│   ├── patching.py             # Patch grid (112 µm) and tissue filtering (≥ 30%)
│   └── phikon_v2_model.py      # Updated extractor saving unpooled patch arrays (M_i x 1024)
└── models/
    ├── abmil.py                # ABMIL architecture, GatedAttention, and Attention Entropy
    ├── mlp.py                  # DistributionalMLP head and GaussianBetaNLLLoss
    ├── torch_cv.py             # Modular PyTorch nested grouped CV harness
    └── evaluation.py           # Patient bootstrap CIs, permutations, and ICC

scripts/model_train/phase2/
└── train_phase2_abmil.py       # Phase 2 runner script

results/phase2_abmil/           # Outputs: cv_predictions.csv, cv_metrics_summary.csv, 
                                # attention_weights.parquet, calibration_summary.csv
```

---

## 3. Architecture & Mathematical Formulation

```
Unpooled Patch Embeddings for Tile i
        H_i = [h_1, h_2, ..., h_M]   ∈  R^(M_i × 1024)
                         │
                         ▼
             Gated Attention Mechanism
       e_m = w^T [ tanh(V h_m) ⊙ sigmoid(U h_m) ]
       a_m = exp(e_m) / ∑_j exp(e_j)
                         │
                         ▼
        Attention Weights: a = [a_1, a_2, ..., a_M]   (∑ a_m = 1)
        ────────────────────────┬─────────────────────────
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
Attention Pooling                              Heterogeneity Analysis
z_tile = ∑ (a_m · h_m) ∈ R^1024                1. Spatial Heatmaps: a_m → (x_m, y_m)
        │                                      2. Attention Entropy: H_tile
        ▼                                      3. Distribution Shape: Sparsity & Gini
Distributional Head
(mu, sigma) for 5 Pathways
```

### A. Gated Attention Mechanism (Ilse et al., ICML 2018)
For each patch $m \in \{1, \dots, M_i\}$ with embedding $h_m \in \mathbb{R}^{d}$ ($d = 1024$):
$$e_m = \mathbf{w}^\top \left[ \tanh(\mathbf{V} h_m) \odot \text{sigmoid}(\mathbf{U} h_m) \right]$$
$$a_m = \frac{\exp(e_m)}{\sum_{j=1}^{M_i} \exp(e_j)}$$

Where:
* $\mathbf{V} \in \mathbb{R}^{L \times d}$ and $\mathbf{U} \in \mathbb{R}^{L \times d}$ are learnable projection matrices (e.g. $L = 128$).
* $\odot$ is element-wise multiplication (the gating mechanism that allows the network to learn non-linear feature interactions).
* $\mathbf{w} \in \mathbb{R}^{L \times 1}$ is the attention scoring vector.
* The softmax ensures that $\sum_{m=1}^{M_i} a_m = 1$ and $a_m > 0$.

### B. Attention Pooling
The pooled representation of the tile is the attention-weighted linear combination of its patch vectors:
$$\mathbf{z}_{\text{tile}} = \sum_{m=1}^{M_i} a_m h_m \quad \in \mathbb{R}^{1024}$$

> [!NOTE]
> **Variable Bag Size Invariance:** No matter whether Tile A has 52 valid patches and Tile B has 81 valid patches, the pooled vector $\mathbf{z}_{\text{tile}}$ is **always strictly 1024-dimensional**. This eliminates the need for padding or awkward fixed-length transformations.

### C. Distributional Prediction Head
The pooled embedding $\mathbf{z}_{\text{tile}}$ passes through LayerNorm and linear projection heads:
$$\mu = \mathbf{W}_\mu \mathbf{z}_{\text{tile}} + \mathbf{b}_\mu \quad \in \mathbb{R}^5$$
$$\log \sigma = \mathbf{W}_\sigma \mathbf{z}_{\text{tile}} + \mathbf{b}_\sigma \quad \in \mathbb{R}^5$$
$$\sigma = \text{clamp}\left(\exp(\log \sigma), \; 10^{-3}, \; 10.0\right)$$

Trained via **Gaussian $\beta$-NLL ($\beta = 0.5$)**:
$$\mathcal{L}_{\beta\text{-NLL}}(\mu, \sigma, y) = \frac{1}{2} \left[ \frac{(y - \mu)^2}{\sigma^2 + \epsilon} + \log(\sigma^2 + \epsilon) \right] \cdot \left(\sigma^2\right)^{\beta}_{\text{detached}}$$

---

## 4. Attention Weights as Intratumoral Heterogeneity Signals

In standard classification, attention weights are often treated as mere interpretability tools. In our project, **the attention distribution itself is a primary biological readout of intratumoral heterogeneity (ITH)**.

### A. Spatial Heatmap Projection
Each patch $m$ has known native coordinates $(x_m, y_m)$ in the full-resolution microscope scan.
* By rendering $a_m$ as a 2D colormap over the original H&E tile, we reconstruct a **sub-tile spatial activation map**.
* Clinicians can immediately visualize whether pathway activity is localized to:
  1. Dense invasive tumor nests vs. stroma.
  2. Tumor-immune interfaces (e.g. tertiary lymphoid structures).
  3. Hypoxic/necrotic cores.

### B. Attention Entropy ($H_{\text{tile}}$)
We quantify the spread of morphological attention across the tile using Shannon Entropy:
$$H_{\text{tile}} = -\sum_{m=1}^{M_i} a_m \log(a_m + \epsilon)$$

To account for slight variations in patch counts between edge tiles ($M_i \approx 55$) and interior tiles ($M_i \approx 81$), we compute the **Normalized Attention Entropy**:
$$H_{\text{norm}} = \frac{H_{\text{tile}}}{\log(M_i)} \quad \in [0, 1]$$

* **High Entropy ($H_{\text{norm}} \to 1.0$):** Attention is uniformly distributed across all patches ($a_m \approx 1/M_i$). Morphological evidence is diffuse throughout the 1 mm tile (homogeneous tumor tissue).
* **Low Entropy ($H_{\text{norm}} \ll 1.0$):** Attention collapses onto a small subset of dominant patches ($a_k \gg 0$). Indicates **focal sub-clonal hotspots**, extreme morphological diversity, or localized pathway activation.

### C. The Shape of the Attention Distribution
The statistical shape of the attention distribution $\{a_1, \dots, a_{M_i}\}$ provides deeper diagnostic insights:

| Distribution Shape Property | Formula / Metric | Biological Interpretation |
|---|---|---|
| **Sparsity / Inequality** | Gini Coefficient of $a$:<br>$G = \frac{\sum_j \sum_k |a_j - a_k|}{2 M_i \sum_j a_j}$ | High Gini indicates strong spatial compartmentalization (a few micro-niches drive the score). |
| **Dominance Ratio** | Top-$k$ Attention Mass:<br>$R_{\text{top-}k} = \sum_{m \in \text{top } k} a_m$ (e.g. $k=5$) | Measures the fraction of the pathway signal concentrated in the top 5 patches (~6% of tile area). |
| **Bimodality / Skewness** | Sample skewness of $\{a_m\}$ | Bimodal attention reveals distinct histological phases (e.g. 30% high-grade tumor islands embedded in 70% non-contributing stroma). |

---

## 5. Experimental Protocol & Fixed Settings

To preserve strict scientific comparability with Phase 1 and Phase 1b, the following choices are locked in the nested CV harness:

| Setting | Value | Rationale |
|---|---|---|
| **Outer Folds** | 5 folds, grouped by patient (`GroupKFold`, Seed 42) | Guarantees identical outer test folds across Phase 1, 1b, and 2. |
| **Inner Folds** | 5 folds, grouped by patient (Seeds 43–47) | Patient-level early stopping without touching outer test patients. |
| **Input Features** | Phikon-v2 unpooled patch embeddings ($M_i \times 1024$) | Direct patch-level representations from 112 µm crops. |
| **Attention Hidden Dim ($L$)** | 128 | Standard capacity for gated attention heads. |
| **Dropout** | 0.25 on attention weights and trunk | Regularization against patient-level memorization. |
| **Optimizer** | AdamW ($\text{lr} = 5 \times 10^{-4}$, weight decay $= 10^{-3}$) | Stable convergence for MIL architectures. |
| **Batch Size** | 16–32 tiles (using `collate_tile_bags`) | Small batch sizes suited for variable patch bag tensors. |
| **Objective** | Gaussian $\beta$-NLL ($\beta = 0.5$) | Prevents outlier-driven uncertainty inflation while predicting $\mu, \sigma$. |
| **Target Normalization** | Cohort z-score on training fold only | Leak-free common scale; test predictions back-transformed to raw units. |

---

## 6. Baselines to Beat

All comparisons are conducted on the exact same 92 patients and 2,993 tiles under patient-level bootstrap (2,000 resamples) and permutation tests (10,000 iterations):

| Model | Architecture | Aggregation | Primary Metric ($r_{\text{within}}$) | Between-Patient $r$ |
|---|---|---|---|---|
| **Phase 1 Baseline** | Ridge Regression | Uniform Mean-Pooling | **$0.29\text{--}0.39$** | $0.11\text{--}0.68$ |
| **Phase 1b Baseline** | Distributional MLP | Uniform Mean-Pooling | **$0.27\text{--}0.34$** | $0.11\text{--}0.70$ |
| **Phase 2 (ABMIL)** | **ABMIL + Distr Head** | **Learned Attention Pooling** | **Target: $> 0.39$** | **Target: $> 0.70$** |

### What constitutes a success in Phase 2?
1. **Beating Phase 1 Ridge on Within-Patient Ranking ($r_{\text{within}}$):** Proves that learned attention over sub-patches successfully recovers the spatial contrast lost by mean-pooling.
2. **Rule 8 Calibration:** Calibrated spatial uncertainty $\sigma$ matching or exceeding Phase 1b coverage ($\approx 60\text{--}68\%$) and positive error-uncertainty correlation ($\rho > 0.25$).
3. **Biological Heterogeneity Validation:** Showing that low Attention Entropy ($H_{\text{norm}}$) correlates with high spatial transcriptomic variance across the true Visium spots inside the tile.

