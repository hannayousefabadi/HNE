"""scripts/additional_signatures/plot_fmrp_variant_maps.py

Runs locally from the pushed spot tables (after add_fmrp_variant_scores.py has been run on the server).
Up and down are in the FMRP-active direction (gene_sets.load_fmrp_variants flips the knockout lists). For the bulk
and the all FMRP signature, the score is ssGSEA(up genes) - ssGSEA(down genes), "up minus down". "all_noAPM" is the
all signature with its APM genes (PSMB8, PSMB10, B2M, all in its down list) removed, so its association with APM
is not partly the same genes.
Writes, in results/additional_signatures/plots/:
  surface_{APM4,APM12}_by_FMRP_{bulk,all,all_noAPM}.png
      reference-style maps: colour = APM score over FMRP (up minus down) x immune count
  fmrp_updown_vs_apm.png
      per patient, rho(FMRP up genes, APM) against rho(FMRP down genes, APM). If FMRP and APM are linked, the two
      parts go opposite ways; if they do not, "up minus down" mixes two different effects and only the parts count
and fmrp_updown_vs_apm_summary.csv / fmrp_updown_vs_apm_per_patient.csv (the numbers of that figure),
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
# composite name -> (up set, down set)
COMPOSITES = {"bulk": ("FMRP_bulk_up", "FMRP_bulk_down"),
              "all": ("FMRP_all_up", "FMRP_all_down"),
              "all_noAPM": ("FMRP_all_up", "FMRP_all_down_noAPM")}

variants = load_fmrp_variants()
patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
needed = [f"{name}_score" for name in variants]
absent = [c for c in needed if c not in spots.columns]
if absent:
    raise KeyError(f"spot tables lack {absent}: run add_fmrp_variant_scores.py on the server and push the tables")
for kind, (up, down) in COMPOSITES.items():
    spots[f"FMRP_{kind}_updown_score"] = spots[f"{up}_score"] - spots[f"{down}_score"]
dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients")

rows = []
scores = {name.removeprefix("FMRP_").replace("_", " "): f"{name}_score" for name in variants}
scores.update({f"{kind} updown": f"FMRP_{kind}_updown_score" for kind in COMPOSITES})
for label, score in scores.items():
    for other in ("immune_count", "APM4_score", "APM12_score"):
        rho = association.per_patient_rho_pair(dominant, score, other, CONFIG["min_spots_per_patient"])
        rows.append({"fmrp": label, "against": other.removesuffix("_score"),
                     **association.patient_level_summary(rho, CONFIG["n_boot"], CONFIG["seed"])})
summary = pd.DataFrame(rows)
summary.to_csv(ADDITIONAL_SIGNATURES / "fmrp_variants_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

for kind in COMPOSITES:
    for apm in ("APM4", "APM12"):
        association.plot_surface(dominant, f"{apm}_score", f"{apm} score",
                                 ADDITIONAL_SIGNATURES_PLOTS / f"surface_{apm}_by_FMRP_{kind}.png", CONFIG,
                                 x_score=f"FMRP_{kind}_updown_score", x_label=f"FMRP {kind} (up minus down)")
per_patient, opposition = association.plot_updown_opposition(
    dominant, {kind: (f"{up}_score", f"{down}_score") for kind, (up, down) in COMPOSITES.items()},
    {"APM4": "APM4_score", "APM12": "APM12_score"},
    ADDITIONAL_SIGNATURES_PLOTS / "fmrp_updown_vs_apm.png",
    CONFIG["min_spots_per_patient"], CONFIG["n_boot"], CONFIG["seed"])
opposition.to_csv(ADDITIONAL_SIGNATURES / "fmrp_updown_vs_apm_summary.csv", index=False)
per_patient.to_csv(ADDITIONAL_SIGNATURES / "fmrp_updown_vs_apm_per_patient.csv", index=False)
print(opposition.round(3).to_string(index=False))
print("plots in", ADDITIONAL_SIGNATURES_PLOTS)
