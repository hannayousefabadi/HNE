"""QC module for spots and signature analysis"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from hne.paths import PREPROCESSING_QC_REPORTS

logger = logging.getLogger(__name__)

class QCTracker:
    """Collects records and save to csv."""
    def __init__(self,
                 mode='single_patient',   # or 'cohort'
                 ):
        """
        Args:
            mode: 'single_patient' or 'cohort' - determines subdirectory
        """
        
        self.output_dir = Path(PREPROCESSING_QC_REPORTS) / mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.records = []
        
    def add_record(self, patient_id, stage, status, message, metadata=None):
        """Add a QC record"""
        timestamp = datetime.now().isoformat()    

        record = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "stage": stage,     # "tumor_purity", "tiling", "signatures", etc.
            "status": status,   # "PASS", "WARNING", "ERROR", "SKIPPED"
            "message": message,
        }
        
        # add metadata if provided
        if metadata:
            record.update(metadata)

        self.records.append(record)

        # use logger for console / file output
        if status == "ERROR":
            logger.error(f"[{patient_id}] {stage}: {message}")
        elif status == "WARNING":
            logger.warning(f"[{patient_id}] {stage}: {message}")
        else:
            logger.info(f"[{patient_id}] {stage}: {message}")      


    def to_dataframe(self):
        """Convert records to dataframe"""
        return pd.DataFrame(self.records)
    
    
    def save_summary(self):
        """Save summary csv and return exclusion list"""
        output_path = self.output_dir / "qc_summary.csv"
        df = self.to_dataframe()

        if len(df) == 0:
            print("No QC records to summarize")
            return pd.DataFrame()
        
        # count issues per patient
        summary = df.groupby('patient_id').agg({
            'status': lambda x: (x == 'ERROR').sum(),
            'patient_id': 'count'
        }).rename(columns={'status': 'n_errors', 'patient_id': 'n_checks'})
        
       # check which patients had tumor tiles
        tumor_filter_results = df[df['stage'] == "tumor_filter"]
        no_tumor_patients = set(tumor_filter_results[tumor_filter_results['status'] == "ERROR"]["patient_id"])

        # flag patients that should be excluded
        summary['exclude'] = (summary['n_errors'] > 0) | (summary.index.isin(no_tumor_patients))
        summary['exclude_reason'] == ""
        summary.loc[summary['n_errors'] > 0, 'exclude_reason'] = "Has errors"
        summary.loc[summary.index.isin(no_tumor_patients), 'exclude_reason'] = "No tumor tiles"
        
        (df[df['stage']=='tumor_filter']['status'] == 'SKIPPED')
        
        
        summary.to_csv(output_path)
        logger.info(f"QC summary saved to {output_path}")
        return summary
    
    
    # def save_aggregated_qc_plots(self, all_patients_data):
        """Generate cohort-level QC plots"""
        # # Combine all patients' tile signatures
        # all_tiles = pd.concat([p['tiles_sig'] for p in all_patients_data if p['tiles_sig'] is not None])
        
        # Generate same plots but for entire cohort
        # signature_variation
        # signature_distribution(all_tiles, self.sig_cols, save_path= output_dir /"cohort_distribution.png")
        # signature_sparsity()
        # signature_consistency()
        # signature_correlation()




def signature_variation(spots_df, 
                        tumor_tiles, 
                        sig_cols, 
                        patient_id=None, 
                        mode='single_patient',
                        save_dir=PREPROCESSING_QC_REPORTS):
    """
    Calculate and plot signature variation across tumor spots

    Args:
        spots_df: DataFrame with spot-level signature
        tumor_tiles: DataFrame with tumor tiles information
        sig_cols: List of signature column names
        patient_id: Optional patient ID for naming saved plots
        save_dir: Dir to save plots, if None plot are displayed

    Returns:
        tumor_spots: Filtered DataFrame with only tumor spots
        variation_df: DataFrame with variation statistics
    """
    tumor_spots = spots_df[spots_df["tile_id"].isin(tumor_tiles["tile_id"])].copy()

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
    plt.title(f"Variation of pathway signatures across tumor spots" + 
              (f" - {patient_id}" is patient_id else ""))
    plt.ylabel("Standard deviation")
    plt.tight_layout()
    
    # save
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"signature_variation_{patient_id.png}" if patient_id else "signature_variation.png"
    plt.savefig(save_dir / filename, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved variation plot to {save_dir / filename}")

    return tumor_spots, variation_df



def signature_distribution(sig_cols, tumor_spots, patient_id=None, mode='single_patient', save_dir=PREPROCESSING_QC_REPORTS):
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
    logger.info(f"Saved signature distribution plot to {save_dir / file_name}")




def signature_sparsity(sig_cols, tumor_spots, patient_id=None, mode='single_patient', save_dir=PREPROCESSING_QC_REPORTS):
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
        palette="viridis"
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
    logger.info(f"Saved sparsity plot to {save_dir / file_name}")

    return sparsity_df        


def signature_consistency(vis, tumor_spots, signature_genes, patient_id=None, mode='single_patient', save_dir=PREPROCESSING_QC_REPORTS):
    """
    Calculate internal consistency of signatures (mean gene-gene correlation)

    Args:
        vis: AnnData object with expression data
        tumor_spots: DataFrame with tumor spots barcodes
        signature_genes: Dic of signature names -> list of gene
        patient_id: Optional patient ID for naming saved plots
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
        palette="coolwarm"
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



def signature_correlation(sig_cols, tumor_spots, patient_id=None, mode='single_patient', save_dir=PREPROCESSING_QC_REPORTS):
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
    logger.info(f"Saved signature correlation plot to {save_dir / file_name}")
    plt.close()
    
    return corr_matrix


