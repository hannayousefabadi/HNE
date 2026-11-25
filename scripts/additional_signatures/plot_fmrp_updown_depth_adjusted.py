"""scripts/additional_signatures/plot_fmrp_updown_depth_adjusted.py

Runs locally from the pushed spot tables (after add_spot_depth.py and add_fmrp_variant_scores.py have been run on
the server). Repeats the per-patient up / down vs APM check of plot_fmrp_variant_maps.py with every Spearman rho
replaced by a partial Spearman rho controlling for spot depth (total_counts and n_genes_detected), because every
ssGSEA score rises with the number of genes detected in a spot. The unadjusted outputs are not touched.
Writes:
  results/additional_signatures/plots/fmrp_updown_vs_apm_depth_adjusted.png
  results/additional_signatures/fmrp_updown_vs_apm_depth_adjusted_summary.csv / ..._per_patient.csv
  results/additional_signatures/score_depth_rho_summary.csv: per-patient Spearman rho of each score with each depth
      column, mean over patients with a patient-bootstrap CI and a sign-flip permutation p
"""
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.analysis_config import CANCER_DOMINANT_THRESHOLD, EXCLUDED_PATIENTS
from hne.additional_signatures.paths import (ADDITIONAL_SIGNATURES, ADDITIONAL_SIGNATURES_PLOTS,
                                             NAIVE_PATIENTS_CSV, SPOT_TABLES)

CONFIG = {
    "min_spots_per_patient": 50,      # same as plot_fmrp_variant_maps.py
    "n_boot": 2000,
    "seed": 0,
}
DEPTH = ["total_counts", "n_genes_detected"]
# composite name -> (up set, down set), as in plot_fmrp_variant_maps.py
COMPOSITES = {"bulk": ("FMRP_bulk_up", "FMRP_bulk_down"),
              "all": ("FMRP_all_up", "FMRP_all_down"),
              "all_noAPM": ("FMRP_all_up", "FMRP_all_down_noAPM")}
APM = {"APM4": "APM4_score", "APM12": "APM12_score"}

patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
scores = sorted({f"{s}_score" for pair in COMPOSITES.values() for s in pair} | set(APM.values()))
absent = [c for c in [*DEPTH, *scores] if c not in spots.columns]
if absent:
    raise KeyError(f"spot tables lack {absent}: run add_spot_depth.py / add_fmrp_variant_scores.py on the server and push the tables")
dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients")

rows = []
for score in [*scores, "immune_count"]:
    for depth in DEPTH:
        rho = association.per_patient_rho_pair(dominant, score, depth, CONFIG["min_spots_per_patient"])
        rows.append({"score": score.removesuffix("_score"), "depth": depth,
                     **association.patient_level_summary(rho, CONFIG["n_boot"], CONFIG["seed"])})
depth_summary = pd.DataFrame(rows)
depth_summary.to_csv(ADDITIONAL_SIGNATURES / "score_depth_rho_summary.csv", index=False)
print(depth_summary.round(3).to_string(index=False))

per_patient, opposition = association.plot_updown_opposition(
    dominant, {kind: (f"{up}_score", f"{down}_score") for kind, (up, down) in COMPOSITES.items()}, APM,
    ADDITIONAL_SIGNATURES_PLOTS / "fmrp_updown_vs_apm_depth_adjusted.png",
    CONFIG["min_spots_per_patient"], CONFIG["n_boot"], CONFIG["seed"], covariates=DEPTH)
opposition.to_csv(ADDITIONAL_SIGNATURES / "fmrp_updown_vs_apm_depth_adjusted_summary.csv", index=False)
per_patient.to_csv(ADDITIONAL_SIGNATURES / "fmrp_updown_vs_apm_depth_adjusted_per_patient.csv", index=False)
print(opposition.round(3).to_string(index=False))
print("plot in", ADDITIONAL_SIGNATURES_PLOTS)
