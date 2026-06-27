import pandas as pd

from hne.utils import setup_logging, get_logger
from hne.core.data_io import save_metadata
from hne.preprocessing.pipeline import preprocess_patient
from hne.preprocessing_qc.tracker import QCTracker
from hne.core.paths import TILE_FEATURES, PREPROCESSING_QC_REPORTS, PATIENT_IDS

setup_logging(
    log_file= PREPROCESSING_QC_REPORTS / "cohort" / "cohort.log", 
    console_level="WARNING",
    file_level="DEBUG",
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger()

if __name__ == "__main__":
    
    logger.info("=" * 40)
    logger.info("Starting cohort preprocessing")
    logger.info("=" * 40)
    
    qc = QCTracker(mode='cohort')
    all_metadata = []
    all_tiles_sig = []
    all_spot_data = []
    sig_cols = None

    for patient_id in PATIENT_IDS:
        try: 
            metadata, tiles_sig, spot_df, sig_cols_patient = preprocess_patient(
                patient_id,
                mode='cohort',
                tile_size=100,        # in pixels, ≈ 1 mm in hires image
                k=2, 
                tumor_threshold=0.3,  # tile at least has 30% tumor purity
                min_spots=40,         # tile at least has 40 spots
                qc_tracker=qc,
                verbose=False,        # console quiet
                run_qc_plots=False
            )
            all_metadata.append(metadata)
            all_spot_data.append(spot_df)
            if tiles_sig is not None:
                all_tiles_sig.append(tiles_sig)  # to do: add all_tiles_sig return for later analysis

            # collect signature columns from first successful patient    
            if sig_cols is None and sig_cols_patient:
                sig_cols = sig_cols_patient   

        except Exception as e:
            logger.exception(
                f"Failed preprocessing patient {patient_id}"
            )
            continue         

    qc.save_qc_records()
    summary = qc.save_summary()
    save_metadata(all_metadata, qc.output_dir / "metadata.csv")
    print("\nQC verdicts:")
    print(summary["verdict"].value_counts())

    # generate cohort-level QC plots (requires collecting sig matrices)
    if sig_cols and all_spot_data:
        qc.save_cohort_spot_qc_plots(all_spot_data, sig_cols)
    else:
        logger.warning("No signature columns or spot data - skipping cohort QC plots")    

    logger.info("=" * 40)
    logger.info(f"Processed {len(all_metadata)} patients")
    logger.info("Cohort preprocessing completed")

