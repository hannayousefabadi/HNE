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
    all_results = []

    for patient_id in PATIENT_IDS:
        metadata, tiles = preprocess_patient(
            patient_id,
            mode='cohort',
            tile_size=100,        # in pixels, ≈ 1 mm in hires image
            k=2, 
            tumor_threshold=0.3,  # tile at least has 30% tumor purity
            min_spots=40,         # tile at least has 40 spots
            qc_tracker=qc,
            verbose=False,          # console output level (True=INFO, False=WARNING)
            run_qc_plots=True
        )
        all_results.append(metadata)

    summary = qc.save_summary()
    print(f"Processed {len(all_results)} patients")

