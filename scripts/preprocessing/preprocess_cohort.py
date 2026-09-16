"""Preprocess cohort"""

import pandas as pd
from tqdm import tqdm
import logging
import json
from dataclasses import asdict

from hne.utils import setup_logging, get_logger
from hne.core.data_io import save_metadata, save_tile_features
from hne.preprocessing.pipeline import preprocess_patient
from hne.preprocessing_qc.tracker import QCTracker
from hne.core.paths import PATIENT_IDS, PREPROCESSING_QC_REPORTS
from hne.preprocessing_qc.plots import cohort_tile_variation, save_cohort_spot_qc_plots
from hne.preprocessing.preprocessing_config import PREPROCESSING_CONFIG

setup_logging(
    log_file= PREPROCESSING_QC_REPORTS / "cohort" / "cohort.log", 
    console_level="WARNING",
    file_level="DEBUG",
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# silencing botto logger
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

logger = get_logger()

if __name__ == "__main__":
    
    logger.info("=" * 40)
    logger.info("Starting cohort preprocessing")
    logger.info("=" * 40)
    
    # print the active central configuration
    cohort_qc_dir = PREPROCESSING_QC_REPORTS / "cohort"
    cohort_qc_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(PREPROCESSING_CONFIG)
    config_save_path = cohort_qc_dir / "run_config.json"
    with open(config_save_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print("\n=== Preprcoessing configurations ===")
    for key, val in config_dict.items():
        print(f"    {key}: {val}")
    print("=" * 40)        

    # load patients from cohort_manifest.json 
    patient_ids = PATIENT_IDS
    logger.info(f"Found {len(patient_ids)} patients from cohort manifest")

    if not patient_ids:
        logger.error("No patients found! check S3 paths and permissions.")
        exit(1)

    logger.info(f"First 10 patients: {patient_ids[:10]}")
    logger.info("-" * 40)    


    qc = QCTracker(mode='cohort')
    all_metadata = []
    all_tiles_sig = []
    all_spot_data = []
    sig_cols = None
    failed_patients = []

    for patient_id in tqdm(patient_ids, desc="Preprocessing patients"):
        try: 
            logger.info(f"Preprocessing patient: {patient_id}")
            metadata, tiles_sig, spot_df, sig_cols_patient = preprocess_patient(
                patient_id,
                mode='cohort',
                cfg=PREPROCESSING_CONFIG,
                qc_tracker=qc,
                verbose=False,        # console quiet
                run_qc_plots=False    # this is per-patient plots
            )

            all_metadata.append(metadata)

            if spot_df is not None:
                all_spot_data.append(spot_df)

            if tiles_sig is not None:
                all_tiles_sig.append(tiles_sig)

            # collect signature columns from first successful patient    
            if sig_cols is None and sig_cols_patient:
                sig_cols = sig_cols_patient   

        except Exception as e:
            logger.exception(
                f"Failed preprocessing patient {patient_id}"
            )
            failed_patients.append(patient_id)
            continue         

    qc.save_qc_records()
    summary = qc.save_summary()
    print("\nQC verdicts:")
    print(summary["verdict"].value_counts())

    if all_metadata:
        save_metadata(all_metadata, PREPROCESSING_QC_REPORTS / "cohort" / "metadata.csv")

    # generate cohort-level QC plots
    if all_tiles_sig:
        cohort_tile_df = pd.concat(all_tiles_sig, ignore_index=True)
        # save to csv
        save_tile_features(all_tiles_sig, mode='cohort')
        logger.info("Saved cohort tile signatures to tile_signature_matrix directory")

        # cohort proof of concept plots
        if sig_cols:
            rates = cohort_tile_variation(cohort_tile_df, sig_cols)
            logger.info(f"Generated cohort tile variation plots. Positivity rates:\n{rates.to_string(index=False)}")

    else:
        logger.warning("No tile signatures generated for the cohort")    

    
    if sig_cols and all_spot_data:
        save_cohort_spot_qc_plots(all_spot_data, sig_cols)
    else:
        logger.warning("No signature columns or spot data - skipping cohort QC plots")    

    logger.info("=" * 40)
    logger.info(f"Total patients discovered: {len(patient_ids)}")
    logger.info(f"Successfully processed: {len(all_metadata)}")
    logger.info(f"Failed patients: {len(failed_patients)}")
    if failed_patients:
        logger.warning(f"Failed patients: {failed_patients}")
        
    logger.info("Cohort preprocessing completed")

