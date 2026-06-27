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
from pathlib import Path
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

        # failed stages
        failed_stages = (
            df[df["status"] != "OK"]
            .groupby("patient_id")["stage"]
            .apply(lambda x: "; ".join(x))
        )

        # QC reasons
        qc_reasons = (
            df[df["status"] != "OK"]
            .groupby("patient_id")["message"]
            .apply(lambda x: " | ".join(x))
        )

        summary = summary.join(failed_stages.rename("failed_stages"))
        summary = summary.join(qc_reasons.rename("qc_reasons"))

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

        summary["failed_stages"] = summary["failed_stages"].fillna("")
        summary["qc_reasons"] = summary["qc_reasons"].fillna("")

        summary = summary.sort_values(
            ["verdict", "n_flags"],
            ascending=[False, False]
        )        
        
        summary.to_csv(output_path)
        return summary    
            