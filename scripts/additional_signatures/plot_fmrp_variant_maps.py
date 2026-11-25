"""scripts/additional_signatures/plot_fmrp_variant_maps.py

Runs locally from the pushed spot tables (after add_fmrp_variant_scores.py has been run on the server).
For the bulk and the all FMRP signature, the score is ssGSEA(up genes) - ssGSEA(down genes), "up minus down".
Writes, in results/additional_signatures/plots/:
  surface_APM4_by_FMRP_bulk.png, surface_APM12_by_FMRP_bulk.png, surface_APM4_by_FMRP_all.png, surface_APM12_by_FMRP_all.png
      reference-style maps: colour = APM score over FMRP (up minus down) x immune count
and fmrp_variants_summary.csv: per-patient Spearman rho of every FMRP component with immune count and with APM4 / APM12,
mean over patients with a patient-bootstrap CI and a sign-flip permutation p.
"""
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.analysis_config import CANCER_DOMINANT_THRESHOLD, EXCLUDED_PATIENTS
from hne.additional_signatures.gene_sets import load_fmrp_variants
from hne.additional_signatures.paths import (ADDITIONAL_SIGNATURES, ADDITIONAL_SIGNATURES_PLOTS,
                                             NAIVE_PATIENTS_CSV, SPOT_TABLES)

CONFIG = {
    "min_spots_per_patient": 50,
    "n_boot": 2000,
    "seed": 0,
    "surface_grid": 80,
    "surface_bandwidth": 0.07,
    "surface_min_support": 8.0,
}
KINDS = ["bulk", "all"]

variants = load_fmrp_variants()
patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
needed = [f"{name}_score" for name in variants]
absent = [c for c in needed if c not in spots.columns]
if absent:
    raise KeyError(f"spot tables lack {absent}: run add_fmrp_variant_scores.py on the server and push the tables")
for kind in KINDS:
    spots[f"FMRP_{kind}_updown_score"] = spots[f"FMRP_{kind}_up_score"] - spots[f"FMRP_{kind}_down_score"]
dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients")

rows = []
for kind in KINDS:
    for part in ("up", "down", "updown"):
        score = f"FMRP_{kind}_{part}_score"
        for other in ("immune_count", "APM4_score", "APM12_score"):
            rho = association.per_patient_rho_pair(dominant, score, other, CONFIG["min_spots_per_patient"])
            rows.append({"fmrp": f"{kind} {part}", "against": other.removesuffix("_score"),
                         **association.patient_level_summary(rho, CONFIG["n_boot"], CONFIG["seed"])})
summary = pd.DataFrame(rows)
summary.to_csv(ADDITIONAL_SIGNATURES / "fmrp_variants_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

for kind in KINDS:
    for apm in ("APM4", "APM12"):
        association.plot_surface(dominant, f"{apm}_score", f"{apm} score",
                                 ADDITIONAL_SIGNATURES_PLOTS / f"surface_{apm}_by_FMRP_{kind}.png", CONFIG,
                                 x_score=f"FMRP_{kind}_updown_score", x_label=f"FMRP {kind} (up minus down)")
print("plots in", ADDITIONAL_SIGNATURES_PLOTS)
