
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from hne.paths import TILES

OUT_DIR = TILES / 

def signature_variation(spots_df, tumor_tiles, sig_cols):
    tumor_spots = spots_df[spots_df["tile_id"].isin(tumor_tiles["tile_id"])]

    variation_df = pd.DataFrame({
        "signature": sig_cols,
        "std": [tumor_spots[col].std() for col in sig_cols],
        "cv":  [tumor_spots[col].std() / np.abs(tumor_spots[col].mean()) for col in sig_cols]
    }).sort_values("std", ascending=False)

    plt.figure(figsize=(8,5))

    sns.barplot(
        data=variation_df,
        x="signature",
        y="std",
        palette="viridis"
    )

    plt.xticks(rotation=45)
    plt.title("Variation of pathway signatures across tumor spots")
    plt.ylabel("Standard deviation")
    plt.show()

    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_fold{fold_idx}.png"
    plt.savefig(fig_path, dpi=150)

    return tumor_spots



def signature_distribution(sig_cols, tumor_spots):
    plt.figure(figsize=(8,6))

    for col in sig_cols:
        sns.kdeplot(tumor_spots[col], label=col, linewidth=2)

    plt.legend()
    plt.title("Distribution of pathway signatures in tumor spots")
    plt.show()



def signature_sparsity(sig_cols, tumor_spots):
    SPARSE_THRESHOLD = 1e-6

    sparsity_df = pd.DataFrame({
        "signature": sig_cols,
        "zero_fraction": [
            (np.abs(tumor_spots[c]) < SPARSE_THRESHOLD).mean()
            for c in sig_cols
        ]
    })
    plt.figure(figsize=(8,5))

    sns.barplot(
        data=sparsity_df,
        x="signature",
        y="zero_fraction",
        palette="viridis"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Fraction of near‑zero scores")
    plt.title("Signature sparsity across tumor spots")
    plt.show()



def signature_consistency(vis, tumor_spots, signature_genes):
    consistency = []
    expr = pd.DataFrame(
        vis.layers["log_norm_count"].toarray(),
        index=vis.obs.index,
        columns=vis.var_names
    )

    expr = expr.loc[tumor_spots["barcode"]]

    for sig, genes in signature_genes.items():
        if len(genes) < 2:
            continue

        gexpr = expr[genes]
        corr = gexpr.corr()

        avg_corr = corr.values[np.triu_indices_from(corr, k=1)].mean()

        consistency.append({
            "signature": sig,
            "mean_gene_corr": avg_corr
        })

    consistency_df = pd.DataFrame(consistency)
    plt.figure(figsize=(7,4))

    sns.barplot(
        data=consistency_df,
        x="signature",
        y="mean_gene_corr",
        palette="coolwarm"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Mean gene correlation")
    plt.title("Internal consistency of pathway signatures")
    plt.show()




def signature_correlation(sig_cols, tumor_spots):
    corr_matrix = tumor_spots[sig_cols].corr()
    plt.figure(figsize=(7,6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation between pathway signatures")
    plt.show()
