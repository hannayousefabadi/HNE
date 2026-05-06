import logging
from tqdm import tqdm
from hne.preprocessing import preprocess_patient
from hne.spots_qc import QCTracker

# Set logging to WARNING for cohort (quiet console)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cohort_preprocessing.log"),  # Save full log
        logging.StreamHandler()                           # Console only shows WARNING+
    ]
)

def preprocess_cohort(patient_ids, tumor_threshold=0.3):
    qc_tracker = QCTracker("cohort_qc_log.csv")
    all_metadata = []
    
    for patient_id in tqdm(patient_ids):
        metadata, tiles = preprocess_patient(
            patient_id,
            tumor_threshold=tumor_threshold,
            qc_tracker=qc_tracker,
            run_qc=False  # No plots for cohort
        )
        all_metadata.append(metadata)
    
    # Generate summary
    summary = qc_tracker.save_summary("cohort_summary.csv")
    print(f"\nProcessed {len(patient_ids)} patients")
    print(f"Excluded: {summary['exclude'].sum()} patients")
    
    return all_metadata

if __name__ == "__main__":
    from hne.paths import PATIENT_IDS
    preprocess_cohort(PATIENT_IDS)






















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