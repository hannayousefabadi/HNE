import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging


from hne.paths import QC_REPORTS


logger = logging.getLogger(__name__)

class QCTracker:
    def __init__(self, output_dir=QC_REPORTS, log_file="preprocessing_errors.log"):
        """
        Initilize QC tracker
        Args:
            output_dir: path to save QC reports and plots
            log_file: name of the error log file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.records = []
        self.log_file = self.output_dir / log_file

        with open(self.log_file, 'w') as f:
            f.write("patient_id, stage, status, message, timestamp\n")

        
    def add_record(self, patient_id, stage, status, message, metadata=None):
        """Add a QC record"""
        timestamp = datetime.now().isoformat()    

        record = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "stage": stage,    # "tumor_purity", "tiling", "signatures", etc.
            "status": status,  # "PASS", "WARNING", "ERROR", "SKIPPED"
            "message": message,
        }
        
        # add metadata if provided
        if metadata:
            record.update(metadata)

        self.records.append(record)

        # also log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{patient_id},{stage},{status},{message},{timestamp}\n")

        # also use logger
        if status == "ERROR":
            logger.error(f"[{patient_id}] {stage}: {message}")

        elif status == "WARNING":
            logger.warning(f"[{patient_id}] {stage}: {message}")

        else:
            logger.info(f"[{patient_id} {stage}: {message}]")      


    def to_dataframe(self):
        """Convert records to dataframe"""
        return pd.DataFrame(self.records)
    
    
    def save_summary(self, output_path="qc_summary.csv"):
        """Save summary csv and return exclusion list"""
        output_path = QC_REPORTS
        



        df = self.to_dataframe()
        
        # Count issues per patient
        summary = df.groupby('patient_id').agg({
            'status': lambda x: (x == 'ERROR').sum(),
            'patient_id': 'count'
        }).rename(columns={'status': 'n_errors', 'patient_id': 'n_checks'})
        
        # Flag patients that should be excluded
        summary['exclude'] = (summary['n_errors'] > 0) | (df[df['stage']=='tumor_filter']['status'] == 'SKIPPED')
        
        summary.to_csv(output_path)
        return summary
    
    
    def save_aggregated_qc_plots(self, all_patients_data, output_dir="cohort_qc_plots"):
        """Generate cohort-level QC plots"""
        # Combine all patients' tile signatures
        all_tiles = pd.concat([p['tiles_sig'] for p in all_patients_data if p['tiles_sig'] is not None])
        
        # Generate same plots but for entire cohort
        signature_distribution(all_tiles, self.sig_cols, save_path=output_dir/"cohort_distribution.png")
        # etc.
    
    




def signature_variation(spots_df, tumor_tiles, sig_cols, logger):
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
