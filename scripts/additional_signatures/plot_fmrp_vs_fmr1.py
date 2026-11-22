"""scripts/additional_signatures/plot_fmrp_vs_fmr1.py

FMRP signature score against FMR1 expression in the cancer-dominant spots of the selected naive patients.
FMR1 is zero-inflated (no count in about half of the spots), so the plot has two parts:
  left   per patient, mean FMRP in spots without vs with an FMR1 count (paired)
  right  in spots with a count only, FMRP against the FMR1 level (log1p CPM), patient-bootstrap band
Writes plots/fmrp_vs_fmr1.png and fmrp_vs_fmr1_summary.csv (patient-level statistics).
Cancer-dominant threshold and excluded patients: hne.additional_signatures.analysis_config.
"""
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.analysis_config import CANCER_DOMINANT_THRESHOLD, EXCLUDED_PATIENTS
from hne.additional_signatures.paths import (ADDITIONAL_SIGNATURES, ADDITIONAL_SIGNATURES_PLOTS,
                                             NAIVE_PATIENTS_CSV, SPOT_TABLES)

CONFIG = {
    "min_spots_per_group": 50,   # a patient needs this many spots in a group (and per rho) to enter the statistics
    "n_bins": 8,                 # FMR1-level quantile bins among spots with a count
    "n_boot": 2000,
    "seed": 0,
}

patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
if "FMR1_expression" not in spots.columns:
    raise KeyError("spot tables have no FMR1_expression, run add_single_gene_columns.py")
dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients; "
      f"FMR1 detected in {(dominant['FMR1_expression'] > 0).mean():.1%}")

stats = association.fmrp_vs_fmr1(dominant, ADDITIONAL_SIGNATURES_PLOTS / "fmrp_vs_fmr1.png", CONFIG["n_bins"],
                                 CONFIG["n_boot"], CONFIG["seed"], CONFIG["min_spots_per_group"])
summary = pd.DataFrame([{"statistic": name, **values} for name, values in stats.items()])
summary.to_csv(ADDITIONAL_SIGNATURES / "fmrp_vs_fmr1_summary.csv", index=False)
print(summary.round(3).to_string(index=False))
