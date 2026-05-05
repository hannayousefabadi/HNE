import logging
import sys
from tqdm import tqdm
import pandas as pd


# Configure logging for cohort (less verbose)
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cohort_preprocessing.log")
    ]
)

from preprocess_single_patient import preprocess_patient
from hne.paths import PATIENT_IDS
from hne.spots_qc import QCTracker


def preprocess_cohort(patient_ids):
    """Process multiple patients and collect summaries"""
    qc_tracker = QCTracker()
    results = []
    
    for patient_id in patient_ids:
        try:
            metadata, tiles = preprocess_patient(
                patient_id, 
                verbose=False,  # Console quiet
                qc_tracker=qc_tracker
            )
            results.append(metadata)
        except Exception as e:
            qc_tracker.add_record(patient_id, "pipeline", "ERROR", str(e))
    
    # Final summary
    summary = qc_tracker.save_summary("cohort_qc.csv")
    print(f"\nExcluded {summary['exclude'].sum()} patients due to QC failures")
    
    return results, qc_tracker



























# def preprocess_cohort(patient_ids, **kwargs):
    
    
#     all_metadata = []
#     failed_patients = []
    
#     for patient_id in tqdm(patient_ids, desc="Processing cohort"):
#         try:
#             # Run with verbose=False to reduce output
#             metadata, tiles = preprocess_patient(patient_id, verbose=False, **kwargs)
#             all_metadata.append(metadata)
            
#             # Quick check after each patient
#             if metadata.get('n_tumor_tiles', 0) == 0:
#                 logging.warning(f"{patient_id}: No tumor tiles")
                
#         except Exception as e:
#             logging.error(f"Failed to process {patient_id}: {e}")
#             failed_patients.append({"patient_id": patient_id, "error": str(e)})
    
#     # Create summary report
#     summary_df = pd.DataFrame(all_metadata)
#     summary_df.to_csv("preprocessing_summary.csv", index=False)
    
#     # Flag problematic patients
#     problems = summary_df[
#         (summary_df['n_tumor_tiles'] < 10) |
#         (summary_df['n_spots_missing_tumor_frac'] > 0)
#     ]
    
#     if len(problems) > 0:
#         print(f"\n⚠️ {len(problems)} patients need review:")
#         print(problems[['patient_id', 'n_tumor_tiles', 'n_spots_missing_tumor_frac']])
    
#     return summary_df, failed_patients


# if __name__ == "__main__":
#     # Run on all patients
#     summary, failed = preprocess_cohort(PATIENT_IDS, tumor_threshold=0.3)