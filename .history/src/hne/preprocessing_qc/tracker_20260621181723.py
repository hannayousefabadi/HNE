"""
QCTracker
├── store QC decisions
├── compute patient verdicts
├── save patient QC
└── save cohort QC summary


Key stages that QCTracker follows:
1) "tumor_fraction"       -> tumor_purity module
2) "tile_purity"          -> tumor_purity module
3) "filter_tumor_tiles"   -> tumor_purity module
4) "signature_qc"         
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
        
    def add_record(self, 
                   patient_id, 
                   stage, 
                   status, 
                   message=None, 
                   metadata=None):
        """Add a QC record per stage for OK, FLAG or EXCLUDE""" 

        record = {
            "patient_id": patient_id,
            "stage": stage,     # "tumor_purity", "tiling", "signatures", etc.
            "status": status,   # "OK", "FLAG", "EXCLUDE"
            "message": message,
        }

        VALID_QC_STAGES = {
            "tumor_fraction",
            "tile_purity",
            "filter_tumor_tiles",
            "signature_qc"
        }

        if stage not in VALID_QC_STAGES:
            raise ValueError(f"Invalid QC stage: {stage}")
        
        VALID_STATUSES = {
            "OK",
            "FLAG",
            "EXCLUDE"
        }

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid QC status: {status}")

        existing = next(
            (
                r for r in self.records
                if r ["patient_id"] == patient_id and r["stage"] == stage
            ),
            None
        )

        if existing is not None:
            self.records.remove(existing)
        
        # add metadata if provided
        if metadata:
            record.update(metadata)

        self.records.append(record)



    def to_dataframe(self):
        return pd.DataFrame(self.records)
    


    def get_patient_verdict(self, patient_id):
        
        patient_records = [
            r for r in self.records
            if r["patient_id"] == patient_id
        ]

        statuses = [r["status"] for r in patient_records]

        if "EXCLUDE" in statuses:
            return "EXCLUDE"
        if "FLAG" in statuses:
            return "REVIEW"
        
        return "OK"



    def save_qc_records(self):
        df = self.to_dataframe()

        df.to_csv(
            self.output_dir / "qc_records.csv",
            index=False
        ) 


    def save_summary(self):
        """Save summary csv and return exclusion list"""
        output_path = self.output_dir / "qc_summary.csv"
        df = self.to_dataframe()

        if len(df) == 0:
            print("No QC records to summarize")
            return pd.DataFrame()
        
        # count issues per patient
        summary = df.groupby('patient_id').agg(
            n_excluded=('status', lambda x: (x == 'EXCLUDE').sum()),
            n_flags=('status', lambda x: (x == 'FLAG').sum()),
            n_ok=('status', lambda x: (x == 'OK').sum()),
            n_stages_evaluated=('patient_id', 'count')
        )

        # patients that should be excluded
        summary["verdict"] = "OK"

        summary.loc[
            summary["n_flags"] > 0,
            "verdict"
        ] = "REVIEW"

        summary.loc[
            summary["n_excluded"] > 0,
            "verdict"
        ] = "EXCLUDE"

        
        summary.to_csv(output_path)
        return summary    
            
    
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
