import logging
import sys
from hne.preprocessing.pipeline import preprocess_patient
from hne.preprocessing_qc.tracker import QCTracker
from hne.utils import setup_logging, get_logger
from hne.paths import RESULTS

setup_logging(
    log_file=RESULTS / "preprocessing" / "logs" / "single_patient.log",
    console_level="INFO",
    file_level="DEBUG",
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = get_logger()

if __name__ == "__main__":
    logger.info("=" * 40)
    logger.info("Starting single patient preprocessing")
    logger.info("=" * 40)

    qc = QCTracker(mode='single_patient')
    
    metadata, tiles_sig, _, __ = preprocess_patient(
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


