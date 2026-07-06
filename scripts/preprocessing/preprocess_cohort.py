"""Preprocess cohort"""

import pandas as pd
from tqdm import tqdm
import logging
from hne.utils import setup_logging, get_logger
from hne.core.data_io import save_metadata, save_tile_features
from hne.preprocessing.pipeline import preprocess_patient
from hne.preprocessing_qc.tracker import QCTracker
from hne.core.paths import PROCESSED_VISIUM_BUCKET, PROCESSED_VISIUM_PREFIX, PREPROCESSING_QC_REPORTS
from hne.preprocessing_qc.plots import cohort_tile_variation, save_cohort_spot_qc_plots

from hne.core.s3_io import S3DataLoader

from hne.core.paths import PATIENT_IDS

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
    

    # dynamic patient S3 patients' name discovery
    # loader = S3DataLoader()

    # patient_ids = loader.list_patients_from_processed(
    #     bucket=PROCESSED_VISIUM_BUCKET,
    #     prefix=PROCESSED_VISIUM_PREFIX
    # )
    
    # logger.info(f"Found {len(patient_ids)} patients in processed data")

    # use the hardcoded list instead
    patient_ids = PATIENT_IDS
    
    logger.info(f"Found {len(patient_ids)} patients in hardcoded list")

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
                target_physical_size_um=1000,       # <-- 1000 µm = 1 mm
                k=2, 
                tumor_threshold=0.3,  # tile at least has 30% tumor purity
                min_spots=40,         # tile at least has 40 spots
                qc_tracker=qc,
                verbose=False,        # console quiet
                run_qc_plots=False,
                use_s3_discovery=False # finding the entire cohort dynamically   
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
        logger.info("Saved cohort tile signatures to tile_features directory")

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

