import glob
from subprocess import run

patients = ["CH_L_282", "CH_L_123", "CH_L_987", ...]

for pid in patients:
    run(["python", "preprocess_single_patient.py", "--patient", pid])
