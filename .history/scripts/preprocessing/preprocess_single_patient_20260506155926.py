
import logging
import sys
from hne.preprocessing import preprocess_patient
from hne.spots_qc import QCTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    qc = QCTracker("single_patient_qc.log")
    metadata, tiles = preprocess_patient(
        "CH_L_282", 
        qc_tracker=qc,
        logger=logger,
        run_qc=True  # Show plots
    )
    print(f"\nMetadata: {metadata}")


