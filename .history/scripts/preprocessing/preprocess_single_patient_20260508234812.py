import logging
import sys
from hne.preprocessing import preprocess_patient
from hne.spots_qc import QCTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("single_patient.log")
    ]
)

if __name__ == "__main__":
    qc = QCTracker(mode='single_patient')
    metadata, tiles = preprocess_patient(
        "CH_L_282", 
        mode='single_patient',
        tile_size=100,        # in pixels, ≈ 1 mm in hires image
        k=2, 
        tumor_threshold=0.3,  # tile at least has 30% tumor purity
        min_spots=40,         # tile at least has 40 spots
        qc_tracker=qc,
        verbose=True,         # console output level (True=INFO, False=WARNING)
        run_qc_plots=True
    )
    qc.save_metadata(metadata)
    print(f"\nMetadata: {metadata}")
    summary = qc.save_summary()
    print(f"\nQC Summary:\n{summary}")


