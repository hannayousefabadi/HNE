
import matplotlib.pyplot as plt
import seaborn as sns


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



def signature_distribution(sig_cols, tumor_spots):
    plt.figure(figsize=(8,6))

    for col in sig_cols:
        sns.kdeplot(tumor_spots[col], label=col, linewidth=2)

    plt.legend()
    plt.title("Distribution of pathway signatures in tumor spots")
    plt.show()



def signature_sparsity():

def signature_consistency():

def signature_correlation():
