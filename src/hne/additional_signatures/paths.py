"""src/hne/additional_signatures/paths.py

Output folders of the additional-signatures analysis (APM, FMRP, immune count in treatment-naive
patients). Kept apart from tile_signature_matrix/ and results/phase*/ on purpose.
"""
from hne.core.config import RESULTS

ADDITIONAL_SIGNATURES = RESULTS / "additional_signatures"
NAIVE_PATIENTS_CSV = ADDITIONAL_SIGNATURES / "naive_patients.csv"
SPOT_TABLES = ADDITIONAL_SIGNATURES / "spot_tables"
ADDITIONAL_SIGNATURES_PLOTS = ADDITIONAL_SIGNATURES / "plots"

for path in [ADDITIONAL_SIGNATURES, SPOT_TABLES, ADDITIONAL_SIGNATURES_PLOTS]:
    path.mkdir(parents=True, exist_ok=True)
