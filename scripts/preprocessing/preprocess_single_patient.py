"""Preprocess single patient"""
import logging
from hne.preprocessing.pipeline import preprocess_patient
from hne.preprocessing_qc.tracker import QCTracker
from hne.utils import setup_logging, get_logger
from hne.core.data_io import save_metadata
from hne.core.paths import PREPROCESSED_SINGLE_PATIENT

setup_logging(
    log_file = PREPROCESSED_SINGLE_PATIENT / "single_patient.log",
    console_level="INFO",
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
    logger.info("Starting single patient preprocessing")
    logger.info("=" * 40)

    qc = QCTracker(mode='single_patient')
    
    metadata, tiles_sig, _, __ = preprocess_patient(
        "CH_L_282a", 
        mode='single_patient',
        target_physical_size_um=1000,     # <-- 1000 µm = 1 mm
        k=2, 
        tumor_threshold=0.3,  # tile at least has 30% tumor purity
        min_spots=40,         # tile at least has 40 spots
        qc_tracker=qc,
        verbose=True,         # console output level (True=INFO, False=WARNING)
        run_qc_plots=True,
        use_s3_discovery=False
    )

    save_metadata(metadata, qc.output_dir / "metadata.csv")
    print(f"\nMetadata: {metadata}")

    summary = qc.save_summary()
    print(f"\nQC Summary:\n{summary}")
    
    logger.info("=" * 40)
    verdict = qc.get_patient_verdict("CH_L_282a")
    logger.info(f"Preprocessing completed | QC verdict={verdict}")
