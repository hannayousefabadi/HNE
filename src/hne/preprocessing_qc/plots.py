"""QC plots for spots signature analysis"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from hne.utils import get_logger

from hne.core.paths import PREPROCESSING_QC_REPORTS

logger = get_logger()

def signature_variation(tumor_spots, 
                        sig_cols, 
                        patient_id=None, 
                        mode='single_patient',
                        save_dir=PREPROCESSING_QC_REPORTS):
    """
    Calculate and plot signature variation across tumor spots

    Args:
        tumor_spots:
        sig_cols: List of signature column names
        patient_id: Optional patient ID for naming saved plots
        save_dir: Dir to save plots, if None plot are displayed

    Returns:
        tumor_spots: Filtered DataFrame with only tumor spots
        variation_df: DataFrame with variation statistics
    """
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
        hue="signature",
        palette="viridis",
        legend=False
    )
    plt.xticks(rotation=45)
    plt.title(f"Variation of pathway signatures across tumor spots" + 
              (f" - {patient_id}" if patient_id else ""))
    plt.ylabel("Standard deviation")
    plt.tight_layout()
    
    # save
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"signature_variation_{patient_id}.png" if patient_id else "signature_variation.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.close()

    return variation_df



def signature_distribution(sig_cols, 
                           tumor_spots, 
                           patient_id=None, 
                           mode='single_patient', 
                           save_dir=PREPROCESSING_QC_REPORTS):
    """Plot distribution of pathway signatures"""
    plt.figure(figsize=(8,6))

    for col in sig_cols:
        sns.kdeplot(tumor_spots[col], label=col, linewidth=2)

    plt.legend()
    plt.title(f"Distribution of pathway signatures in tumor spots" +
              (f" - {patient_id}" if patient_id else ""))
    plt.xlabel("Signature score")
    plt.ylabel("Density")
    plt.tight_layout()

    # save
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"signature_distribution_{patient_id}.png" if patient_id else "signature_distribution.png"
    plt.savefig(save_dir / file_name, dpi=600, bbox_inches="tight")
    plt.close()




def signature_sparsity(sig_cols,
                       tumor_spots, 
                       patient_id=None, 
                       mode='single_patient', 
                       save_dir=PREPROCESSING_QC_REPORTS):
    """Calculate and plot signature sparsity (fraction of non-zero scores)"""
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
        hue="signature",
        palette="viridis",
        legend=False
    )
    plt.xticks(rotation=45)
    plt.ylabel("Fraction of near‑zero scores")
    plt.title(f"Signature sparsity across tumor spots" +
              (f" - {patient_id}" if patient_id else ""))
    plt.ylim(0, 1)
    plt.tight_layout()
    
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"signature_sparsity_{patient_id}.png" if patient_id else "signature_sparsity.png"
    plt.savefig(save_dir / file_name, dpi=600, bbox_inches="tight")
    plt.close()

    return sparsity_df        


def signature_consistency(vis, 
                          tumor_spots, 
                          signature_genes, 
                          patient_id=None, 
                          mode='single_patient', 
                          save_dir=PREPROCESSING_QC_REPORTS):
    """
    Calculate internal consistency of signatures (mean gene-gene correlation)

    Args:
        vis: AnnData object with expression data
        tumor_spots: DataFrame with tumor spots barcodes
        signature_genes: Dic of signature names -> list of gene
        patient_id: Optional patient ID for naming saved plots
        mode: 
        save_dir: Directory to save plots

    """
    consistency = []

    # get the expression matrix
    expr = pd.DataFrame(
        vis.layers["log_norm_count"].toarray(),
        index=vis.obs.index,
        columns=vis.var_names
    )

    expr = expr.loc[tumor_spots["barcode"]]

    for sig, genes in signature_genes.items():
        if len(genes) < 2:
            logger.warning(f"{sig} has < 2 genes, skipping consistency check")
            continue

        # check wich genes are actually in the data
        genes_present = [g for g in genes if g in expr.columns]
        if len(genes_present) < 2:
            logger.warning(f"{sig} has < 2 genes present in the data, skipping")
            continue


        gexpr = expr[genes_present]
        corr = gexpr.corr()

        avg_corr = corr.values[np.triu_indices_from(corr, k=1)].mean()

        consistency.append({
            "signature": sig,
            "mean_gene_corr": avg_corr,
            "n_genes": len(genes_present)
        })

    consistency_df = pd.DataFrame(consistency)

    if len(consistency_df) == 0:
        logger.warning("No consistency metrics could be computed")
        return consistency_df

    plt.figure(figsize=(7,4))
    sns.barplot(
        data=consistency_df,
        x="signature",
        y="mean_gene_corr",
        hue="signature",
        palette="coolwarm",
        legend=False
    )
    plt.xticks(rotation=45)
    plt.ylabel("Mean gene correlation")
    plt.title(f"Internal consistency of pathway signatures" +
              (f" - {patient_id}" if patient_id else ""))
    plt.tight_layout()

    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"signature_consistency_{patient_id}.png" if patient_id else "signature_consistency.png"
    plt.savefig(save_dir / file_name, dpi=600, bbox_inches='tight')
    plt.close()

    return consistency_df



def signature_correlation(sig_cols, 
                          tumor_spots, 
                          patient_id=None, 
                          mode='single_patient', 
                          save_dir=PREPROCESSING_QC_REPORTS):
    """Plot correlation matrix between sigantures"""
    corr_matrix = tumor_spots[sig_cols].corr()

    plt.figure(figsize=(7,6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        fmt='.2f'
    )
    plt.title(f"Correlation between pathway signatures" +
              (f" - {patient_id}" if patient_id else ""))
    plt.tight_layout()

    
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"signature_correlation_{patient_id}.png" if patient_id else "signature_correlation.png"
    plt.savefig(save_dir / file_name, dpi=600, bbox_inches='tight')
    plt.close()
    
    return corr_matrix



def save_cohort_spot_qc_plots(all_spot_data, sig_cols, save_dir=PREPROCESSING_QC_REPORTS):
    """
    Generate cohort-level spot QC plots by aggregating all patients' spots
    
    Args:
        all_spot_data: List of spot DataFrames from each patient (spots_df)
        sig_cols: List of signature column names
        save_dir: Base dir for reports
    """
    if not all_spot_data:
        logger.warning("No spot data provided for cohort spots QC plots.")
        return
    
    output_dir = Path(save_dir) / 'cohort'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # combine all spots from all patients
    all_spots_combined = pd.concat(all_spot_data, ignore_index=True)
    
    # spot-level variation
    variation_df = pd.DataFrame({
        "signature": sig_cols,
        "std": [all_spots_combined[col].std() for col in sig_cols],
        "cv": [all_spots_combined[col].std() / np.abs(all_spots_combined[col].mean()) for col in sig_cols]
    }).sort_values("std", ascending=False)
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=variation_df, x="signature", y="std", hue="signature", palette="viridis", legend=False)
    plt.xticks(rotation=45)
    plt.title("Cohort-level signature variation across all tumor spots")
    plt.ylabel("Standard deviation")
    plt.tight_layout()
    plt.savefig(output_dir / "spot_signature_variation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # spot-level distribution
    plt.figure(figsize=(8,6))
    for col in sig_cols:
        sns.kdeplot(all_spots_combined[col], label=col, linewidth=2)
    plt.legend()
    plt.title("Cohort-level distribution of pathway signatures (spots)")
    plt.xlabel("Signature score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_dir / "spot_signature_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # spot-level sparsity
    SPARSE_THRESHOLD = 1e-6
    sparsity_df = pd.DataFrame({
        "signature": sig_cols,
        "zero_fraction": [
            (np.abs(all_spots_combined[c]) < SPARSE_THRESHOLD).mean()
            for c in sig_cols
        ]
    })
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=sparsity_df, x="signature", y="zero_fraction", hue="signature", palette="viridis", legend=False)
    plt.xticks(rotation=45)
    plt.ylabel("Fraction of near‑zero scores")
    plt.title("Cohort-level signature sparsity across tumor spots")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "spot_signature_sparsity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # spot-level correlation
    corr_matrix = all_spots_combined[sig_cols].corr()
    plt.figure(figsize=(7,6))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt='.2f')
    plt.title("Cohort-level correlation between pathway signatures (spots)")
    plt.tight_layout()
    plt.savefig(output_dir / "spot_signature_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()




def cohort_tile_variation(cohort_tiles_df, sig_cols, save_dir=PREPROCESSING_QC_REPORTS):
    """
    Generate cohort level QC plots specifically for MIL tile features.
    Checks continuous z-score distributions and binary positivity rates.
    """
    save_dir = Path(save_dir) / 'cohort'
    save_dir.mkdir(parents=True, exist_ok=True)

    # A) plot z-score distribution (Violin plot)
    z_cols = [f"{col}_z" for col in sig_cols]

    # melt df for seaborn plotting
    melted_z = cohort_tiles_df.melt(value_vars=z_cols, var_name='Signature', value_name="Z-Score")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=melted_z, x='Signature', y='Z-Score', palette="muted")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Cohort-level distribution of tile z-scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "cohort_tile_zscore_distibution.png", dpi=600)
    plt.close()

    # B) plot binary positivity rates (bar plot)
    binary_cols = [f"{col}_binary" for col in sig_cols]

    # calculate the percentage of tiles that are "1" for each signature
    positivity_rates = (cohort_tiles_df[binary_cols].mean() * 100).reset_index()
    positivity_rates.columns = ["Signature", "Percentage Positive"]
    positivity_rates["Signature"] = positivity_rates["Signature"].str.replace("_binary", "")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=positivity_rates, x="Signature", y="Percentage Positive", palette="viridis")
    plt.title("Percentage of Positive Tiles per Signature across Cohort")
    plt.ylabel("% Positive Tiles (Score > Threshold)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # add the text percentages on top of the bars
    for index, row in positivity_rates.iterrows():
        plt.text(index, row["Percentage Positive"] + 1, f'{row["Percentage Positive"]:.1f}%', 
                 color='black', ha="center")
                 
    plt.tight_layout()
    plt.savefig(save_dir / "cohort_tile_binary_rates.png", dpi=300)
    plt.close()
    
    return positivity_rates


