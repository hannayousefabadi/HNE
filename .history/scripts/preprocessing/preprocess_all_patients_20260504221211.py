
import pandas as pd
from tqdm import tqdm

from hne.paths import PATIENTS
from hne.data_io import load_visium, load_spots, load_scale_factor, load_he_image


for pid, paths in PATIENTS.items():
    adata = load_visium(paths)
    spots = load_spots(paths)
    scale = load_scale_factor(paths)
    img = load_he_image(paths)


# In preprocess_cohort.py


def preprocess_cohort(patient_ids, **kwargs):
    all_metadata = []
    failed_patients = []
    
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        try:
            metadata, ... = preprocess_patient(patient_id, **kwargs)
            all_metadata.append(metadata)
            
        except Exception as e:
            logger.error(f"Failed to process {patient_id}: {e}")
            failed_patients.append({"patient_id": patient_id, "error": str(e)})
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_metadata)
    
    # Save summary report
    summary_df.to_csv("preprocessing_summary.csv", index=False)
    
    # Generate warning flags for problematic patients
    warnings = summary_df[
        (summary_df['n_tumor_tiles'] < 10) |  # Too few tiles
        (summary_df['n_spots_missing_tumor_frac'] > 0) |  # Missing data
        (summary_df['tiles_before_filter'] == 0)  # No tiles at all
    ]
    
    if len(warnings) > 0:
        logger.warning(f"\n⚠️ {len(warnings)} patients need review:")
        print(warnings[['patient_id', 'n_tumor_tiles', 'n_spots_missing_tumor_frac']])
    
    # Save failed patients
    if failed_patients:
        pd.DataFrame(failed_patients).to_csv("failed_patients.csv", index=False)
    
    return summary_df, failed_patients

