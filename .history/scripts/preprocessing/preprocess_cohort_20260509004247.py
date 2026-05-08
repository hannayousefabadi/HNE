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

from preprocessing import preprocess_patient
from hne.spots_qc import QCTracker
from hne.paths import PATIENT_IDS

if __name__ == "__main__":
    
    qc = QCTracker(mode='cohort')
    all_metadata = []
    all_tiles_sig = []
    all_spot_data = []

    for patient_id in PATIENT_IDS:
        metadata, tiles_sig, spot_df = preprocess_patient(
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

    qc.save_summary()
    qc.save_metadata(all_metadata)

    # generate cohort-level QC plots (requires collecting sig matrices)
    sig_cols = [col for col in tiles_sig.columns if col.endswith('_score')]
    qc.save_cohort_tile_qc_plots(all_spot_data, sig_cols)

    print(f"Processed {len(all_metadata)} patients")

