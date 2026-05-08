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
        mode='single_patient'
        qc_tracker=qc,
        verbose=True
    )
    print(f"\nMetadata: {metadata}")
    summary = qc.save_summary()
    print(f"\nQC Summary:\n{summary}")


