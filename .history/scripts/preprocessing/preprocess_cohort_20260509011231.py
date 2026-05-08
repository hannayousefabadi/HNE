import logging
import sys

logging.basicConfig(
    level=logging.WARNING,  # Only warnings in console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cohort_preprocessing.log")
    ]
)

logger = logging.getLogger(__name__) 

from hne.preprocessing import preprocess_patient
from hne.spots_qc import QCTracker
from hne.core.paths import PATIENT_IDS

if __name__ == "__main__":
    
    qc = QCTracker(mode='cohort')
    all_metadata = []
    all_tiles_sig = []
    all_spot_data = []
    sig_cols = None

    for patient_id in PATIENT_IDS:
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
            all_tiles_sig.append(tiles_sig)

        # collect signature columns from first successful patient    
        if sig_cols is None and sig_cols_patient:
            sig_cols = sig_cols_patient    

    qc.save_summary()
    qc.save_metadata(all_metadata)

    # generate cohort-level QC plots (requires collecting sig matrices)
    if sig_cols and all_spot_data:
        qc.save_cohort_spot_qc_plots(all_spot_data, sig_cols)
    else:
        logger.warning("No signature columns or spot data - skipping cohort QC plots")    

    print(f"Processed {len(all_metadata)} patients")

