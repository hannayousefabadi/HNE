import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from hne.utils import get_logger
from hne.core.paths import PREPROCESSING_QC_REPORTS

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
        self.logger = get_logger()
        
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

        # use logger for console and file output
        if status == "ERROR":
            self.logger.error(f"[{patient_id}] {stage}: {message}")
        elif status == "WARNING":
            self.logger.warning(f"[{patient_id}] {stage}: {message}")
        else:
            self.logger.info(f"[{patient_id}] {stage}: {message}")      


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
        summary = df.groupby('patient_id').agg(
            n_errors=('status', lambda x: (x == 'ERROR').sum()),
            n_warnings=('status', lambda x: (x == 'WARNING').sum()),
            n_checks=('patient_id', 'count')
        )
        # n_error: number of errors for this patient, n_checks: number of QC records for this patient
        
       # check which patients had tumor tiles
        tumor_filter_results = df[df['stage'] == "tumor_filter"]
        no_tumor_patients = set(tumor_filter_results[tumor_filter_results['status'] == "ERROR"]["patient_id"])

        # flag patients that should be excluded
        summary['exclude'] = (summary['n_errors'] > 0) | (summary.index.isin(no_tumor_patients))
        summary['exclude_reason'] = ""
        summary.loc[summary['n_errors'] > 0, 'exclude_reason'] = "Has errors"
        summary.loc[summary.index.isin(no_tumor_patients), 'exclude_reason'] = "No tumor tiles"
        
        summary.to_csv(output_path)
        return summary
    
    def save_metadata(self, metadata, output_name="metadata.csv"):
        """Save metadata dict into a single csv"""
        if not metadata:
            logger.warning("No metadata to save")
            return
        
        # handle both a single dict and a list of dicts (single patient vs. cohort)
        if isinstance(metadata, dict):
            metadata = [metadata]

        for item in metadata:
            for key, value in item.items():
                if isinstance(value, set):
                    item[key] = sorted(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, set):
                            value[subkey] = sorted(subvalue)

        metadata_df = pd.DataFrame(metadata)

        # Flatten nested dictionaries if any
        for col in metadata_df.columns:
            if metadata_df[col].apply(lambda x: isinstance(x, dict)).any():
                # Expand nested dicts into separate columns
                expanded = metadata_df[col].apply(pd.Series)
                expanded = expanded.add_prefix(f"{col}_")
                metadata_df = metadata_df.drop(columns=[col]).join(expanded)

        output_path = self.output_dir / output_name
        metadata_df.to_csv(output_path, index=False)
        
        return metadata_df
    
    def save_cohort_spot_qc_plots(self, all_spot_data, sig_cols):
        """
        Generate cohort-level spot QC plots by aggregating all patients' spots
        
        Args:
            all_spot_data: List of spot DataFrames from each patient (spots_df)
            sig_cols: List of signature column names
        """
        # Combine all spots from all patients
        all_spots_combined = pd.concat(all_spot_data, ignore_index=True)
        
        # Spot-level variation
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
        plt.savefig(self.output_dir / "spot_signature_variation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Spot-level distribution
        plt.figure(figsize=(8,6))
        for col in sig_cols:
            sns.kdeplot(all_spots_combined[col], label=col, linewidth=2)
        plt.legend()
        plt.title("Cohort-level distribution of pathway signatures (spots)")
        plt.xlabel("Signature score")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spot_signature_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Spot-level sparsity
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
        plt.savefig(self.output_dir / "spot_signature_sparsity.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Spot-level correlation
        corr_matrix = all_spots_combined[sig_cols].corr()
        plt.figure(figsize=(7,6))
        sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt='.2f')
        plt.title("Cohort-level correlation between pathway signatures (spots)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "spot_signature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
