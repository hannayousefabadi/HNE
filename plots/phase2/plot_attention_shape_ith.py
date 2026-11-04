"""
plots/phase2/plot_attention_shape_ith.py

Generates publication-quality figure analyzing the shape of the Phase 2 ABMIL
attention distribution and connecting attention morphology to Intratumoral Heterogeneity (ITH).

Panels:
  A. Cohort Attention Landscape: Normalized Attention Entropy (H_norm) vs Gini Coefficient.
  B. Attention Lorenz Curves: Quantifying deviation from Phase 1 uniform mean-pooling.
  C. Patch Attention Rank-Decay Curves: Attentional sparsity across focal vs diffuse tiles.
  D. Biological Link to ITH: Morphological attention concentration vs tumor cellularity & patient ITH.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hne.core.paths import PLOTS, RESULTS, TILES_SIGNATURE_MATRIX

# Styling matching publication standards
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.2,
})

OUTPUT_DIR = PLOTS / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTPUT_DIR / "phase2_attention_distribution_ith.png"

DIAG_CSV = RESULTS / "phase2_abmil" / "attention_diagnostics.csv"
ATTN_NPZ = RESULTS / "phase2_abmil" / "attention_weights.npz"


def load_all_data():
    if not DIAG_CSV.exists() or not ATTN_NPZ.exists():
        raise FileNotFoundError("Phase 2 ABMIL results not found. Run scripts/model_train/phase2/train_phase2_abmil.py first.")

    diag_df = pd.read_csv(DIAG_CSV)
    attn_data = np.load(ATTN_NPZ)

    # Load tile metadata (purity and n_spots)
    tiles_dfs = []
    for f in TILES_SIGNATURE_MATRIX.glob("tiles_signature_matrix_*.csv"):
        tiles_dfs.append(pd.read_csv(f)[["patient_id", "tile_id", "tile_purity", "n_spots", "Cell_cycle_signature_score"]])
    tiles_df = pd.concat(tiles_dfs, ignore_index=True).drop_duplicates(subset=["patient_id", "tile_id"])
    merged = diag_df.merge(tiles_df, on=["patient_id", "tile_id"])
    return merged, attn_data


def generate_plot():
    df, attn_data = load_all_data()
    print(f"Loaded {len(df)} tiles for attention shape and ITH visualization.")

    fig = plt.figure(figsize=(15.0, 11.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ==============================================================
    # PANEL A: Cohort Attention Landscape (H_norm vs Gini)
    # ==============================================================
    ax_a.set_title(r"$\mathbf{A.}$  Cohort Attention Morphology Landscape (N = 2,993 Tiles)", weight="bold", loc="left", pad=12)

    sc = ax_a.scatter(
        df["gini_coefficient"],
        df["normalized_attention_entropy"],
        c=df["topk_attention_mass"] * 100.0,
        cmap="viridis",
        alpha=0.65,
        s=22,
        edgecolor="none",
    )
    cbar = plt.colorbar(sc, ax=ax_a, pad=0.03, aspect=25)
    cbar.set_label("Top-5 Patch Attention Mass (%)", fontsize=9.5, weight="bold")

    # Annotate morphological regimes
    ax_a.axvline(0.20, color="#666666", linestyle=":", linewidth=1.1)
    ax_a.axvline(0.45, color="#666666", linestyle=":", linewidth=1.1)

    ax_a.text(0.10, 0.88, "Diffuse Attention\n(Homogeneous Tumor)", ha="center", va="center",
              fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6f5ff", edgecolor="#2b5c8f", alpha=0.9))
    ax_a.text(0.58, 0.88, "Focal Hotspots\n(Compartmentalized ITH)", ha="center", va="center",
              fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2e6", edgecolor="#e66101", alpha=0.9))

    ax_a.set_xlabel("Attention Gini Coefficient (Spatial Inequality)", weight="bold")
    ax_a.set_ylabel("Normalized Attention Entropy ($H_{norm}$)", weight="bold")
    ax_a.set_xlim(0.05, 0.78)
    ax_a.set_ylim(0.82, 1.005)
    ax_a.grid(True, linestyle="--", alpha=0.4)

    # ==============================================================
    # PANEL B: Attention Lorenz Curves vs Uniform Pooling Baseline
    # ==============================================================
    ax_b.set_title(r"$\mathbf{B.}$  Attention Lorenz Curves: Deviation from Mean-Pooling", weight="bold", loc="left", pad=12)

    # 45 degree perfect equality line
    p_grid = np.linspace(0, 1.0, 100)
    ax_b.plot(p_grid * 100, p_grid * 100, color="#222222", linestyle="--", linewidth=1.6, label="Phase 1 Uniform Mean-Pooling (Gini = 0)")

    # Stratify tiles into top 10% (focal), median 10%, bottom 10% (diffuse) by Gini
    q10 = df["gini_coefficient"].quantile(0.10)
    q50_low = df["gini_coefficient"].quantile(0.45)
    q50_high = df["gini_coefficient"].quantile(0.55)
    q90 = df["gini_coefficient"].quantile(0.90)

    focal_tiles = df[df["gini_coefficient"] >= q90]
    median_tiles = df[(df["gini_coefficient"] >= q50_low) & (df["gini_coefficient"] <= q50_high)]
    diffuse_tiles = df[df["gini_coefficient"] <= q10]

    def compute_mean_lorenz(tile_sub):
        lorenz_curves = []
        for _, row in tile_sub.iterrows():
            key = f"{row['patient_id']}_{row['tile_id']}"
            if key not in attn_data:
                continue
            w = np.sort(attn_data[key].astype(np.float64))
            cum_w = np.concatenate([[0.0], np.cumsum(w)])
            cum_p = np.linspace(0.0, 1.0, len(cum_w))
            # Interpolate onto common p_grid
            interp_lorenz = np.interp(p_grid, cum_p, cum_w)
            lorenz_curves.append(interp_lorenz)
        return np.mean(lorenz_curves, axis=0), np.std(lorenz_curves, axis=0)

    lorenz_diffuse, _ = compute_mean_lorenz(diffuse_tiles)
    lorenz_median, _ = compute_mean_lorenz(median_tiles)
    lorenz_focal, _ = compute_mean_lorenz(focal_tiles)

    ax_b.plot(p_grid * 100, lorenz_diffuse * 100, color="#2b5c8f", linewidth=2.0, label=f"Diffuse Tiles (Gini ≤ {q10:.2f})")
    ax_b.plot(p_grid * 100, lorenz_median * 100, color="#2ca02c", linewidth=2.0, label=f"Median Tiles (Gini ≈ {df['gini_coefficient'].median():.2f})")
    ax_b.plot(p_grid * 100, lorenz_focal * 100, color="#d95f02", linewidth=2.2, label=f"Focal Hotspot Tiles (Gini ≥ {q90:.2f})")

    # Fill area for focal inequality
    ax_b.fill_between(p_grid * 100, p_grid * 100, lorenz_focal * 100, color="#d95f02", alpha=0.15, label="Attentive Sparsity Mass")

    ax_b.set_xlabel("Cumulative Proportion of Sub-Patches (%)", weight="bold")
    ax_b.set_ylabel("Cumulative Attention Mass (%)", weight="bold")
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.6)
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # ==============================================================
    # PANEL C: Patch Attention Rank-Decay Curves (Scree Profiles)
    # ==============================================================
    ax_c.set_title(r"$\mathbf{C.}$  Patch Attention Rank-Decay Curves (Sorted Sparsity Profile)", weight="bold", loc="left", pad=12)

    rank_grid = np.arange(1, 81)

    def compute_mean_scree(tile_sub):
        scree_curves = []
        for _, row in tile_sub.iterrows():
            key = f"{row['patient_id']}_{row['tile_id']}"
            if key not in attn_data:
                continue
            w = np.sort(attn_data[key].astype(np.float64))[::-1]  # descending
            # normalize x to 1..80
            x_norm = np.linspace(1, 80, len(w))
            interp_w = np.interp(rank_grid, x_norm, w)
            scree_curves.append(interp_w)
        return np.mean(scree_curves, axis=0), np.std(scree_curves, axis=0)

    scree_diffuse_m, scree_diffuse_s = compute_mean_scree(diffuse_tiles)
    scree_median_m, scree_median_s = compute_mean_scree(median_tiles)
    scree_focal_m, scree_focal_s = compute_mean_scree(focal_tiles)

    ax_c.axhline(1.0 / 80.0, color="#222222", linestyle="--", linewidth=1.5, label="Uniform Weight (1/M ≈ 0.0125)")
    ax_c.plot(rank_grid, scree_focal_m, color="#d95f02", linewidth=2.2, label="Focal Hotspot Profile (Top 10% Gini)")
    ax_c.fill_between(rank_grid, scree_focal_m - scree_focal_s, scree_focal_m + scree_focal_s, color="#d95f02", alpha=0.15)

    ax_c.plot(rank_grid, scree_median_m, color="#2ca02c", linewidth=2.0, label="Median Tile Profile")
    ax_c.plot(rank_grid, scree_diffuse_m, color="#2b5c8f", linewidth=2.0, label="Diffuse Profile (Bottom 10% Gini)")

    ax_c.set_xlabel("Patch Rank (Sorted by Attention Weight)", weight="bold")
    ax_c.set_ylabel("Attention Weight ($a_m$)", weight="bold")
    ax_c.set_xlim(1, 80)
    ax_c.set_ylim(0.0, 0.12)
    ax_c.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.6)
    ax_c.grid(True, linestyle="--", alpha=0.4)

    # Annotate top 5 dominance
    ax_c.annotate("Dominant Micro-Niches\n(Top ~5 patches)", xy=(5, 0.08), xytext=(18, 0.095),
                 arrowprops=dict(facecolor="#d95f02", arrowstyle="->", lw=1.3),
                 fontsize=8.5, weight="bold", color="#d95f02")

    # ==============================================================
    # PANEL D: Linking Morphological Attention to Intratumoral Heterogeneity (ITH)
    # ==============================================================
    ax_d.set_title(r"$\mathbf{D.}$  Connecting Attention Shape to Intratumoral Heterogeneity", weight="bold", loc="left", pad=12)

    # Inset or 2 sub-analyses:
    # 1. Tile purity vs Gini (micro-heterogeneity: tumor-stroma compartmentalization)
    # 2. Patient-level purity spread vs Patient-level Attention Entropy variability
    purity_bins = [0.30, 0.45, 0.60, 0.75, 1.00]
    bin_labels = ["30-45%\n(High Stroma)", "45-60%\n(Mixed)", "60-75%\n(Predom Tumor)", "75-100%\n(Pure Parench)"]
    df["purity_tier"] = pd.cut(df["tile_purity"], bins=purity_bins, labels=bin_labels)

    tier_ginis = [df[df["purity_tier"] == label]["gini_coefficient"].dropna().values for label in bin_labels]

    positions = np.arange(len(bin_labels))
    box = ax_d.boxplot(
        tier_ginis,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="red", markersize=5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = ["#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")

    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(bin_labels, weight="bold")
    ax_d.set_xlabel("Tile Tumor Cellularity / Purity Tier", weight="bold")
    ax_d.set_ylabel("Attention Gini Coefficient ($G$)", weight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4)

    # Correlation annotation
    rho, pval = spearmanr(df["tile_purity"], df["gini_coefficient"])
    ax_d.text(0.96, 0.94, f"Micro-ITH Trend:\nPurity vs Gini Spearman $\\rho = {rho:+.3f}$\n$p = {pval:.1e}$",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=8.8, weight="bold",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#888888", alpha=0.95))

    # Add Patient-level macro-ITH inset
    patient_ith = df.groupby("patient_id").agg(
        std_h_norm=("normalized_attention_entropy", "std"),
        purity_ith=("tile_purity", "std")
    ).dropna()
    r_pt, p_pt = spearmanr(patient_ith["purity_ith"], patient_ith["std_h_norm"])

    ax_inset = ax_d.inset_axes([0.56, 0.16, 0.40, 0.38])
    ax_inset.scatter(patient_ith["purity_ith"], patient_ith["std_h_norm"], s=16, color="#4393c3", alpha=0.8, edgecolor="none")
    # Linear fit line
    m_fit, b_fit = np.polyfit(patient_ith["purity_ith"], patient_ith["std_h_norm"], 1)
    x_line = np.linspace(patient_ith["purity_ith"].min(), patient_ith["purity_ith"].max(), 50)
    ax_inset.plot(x_line, m_fit * x_line + b_fit, color="#b2182b", linewidth=1.4)

    ax_inset.set_title(f"Patient Macro-ITH (r = {r_pt:+.2f}, p = {p_pt:.3f})", fontsize=7.2, weight="bold")
    ax_inset.set_xlabel("Purity Spread ($\\sigma_{purity}$)", fontsize=6.8)
    ax_inset.set_ylabel("Entropy Spread ($\\sigma_{H}$)", fontsize=6.8)
    ax_inset.tick_params(axis="both", labelsize=6.2)
    ax_inset.grid(True, linestyle=":", alpha=0.4)

    # Final save
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully generated Phase 2 attention & ITH figure at: {OUT_PATH}")


if __name__ == "__main__":
    generate_plot()
