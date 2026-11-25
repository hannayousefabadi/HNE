"""scripts/additional_signatures/plot_fmrp_module_scores.py

Runs locally from the pushed spot tables (after add_fmrp_module_scores.py and add_spot_depth.py have been run on the
server). The FMRP up / down analysis of plot_fmrp_variant_maps.py and plot_fmrp_updown_depth_adjusted.py, redone with
Seurat-style module scores instead of ssGSEA for every set (FMRP variants, APM4, APM12). Composite per signature:
module(up) - module(down). The ssGSEA outputs are not touched; every file here has "module" in its name.
Writes, in results/additional_signatures/plots/:
  surface_{APM4,APM12}_module_by_FMRP_{bulk,all,all_noAPM}_module.png
      colour = APM module score over FMRP module (up minus down) x immune count
  fmrp_updown_vs_apm_module.png, fmrp_updown_vs_apm_module_depth_adjusted.png
      per patient, rho of the FMRP up set and of the down set with APM; the second controls for spot depth
      (total_counts, n_genes_detected) with a partial Spearman rho
and in results/additional_signatures/: the numbers of those figures (fmrp_updown_vs_apm_module[_depth_adjusted]
_summary.csv / _per_patient.csv) and module_score_depth_rho_summary.csv (each module score against each depth column).
"""
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.analysis_config import CANCER_DOMINANT_THRESHOLD, EXCLUDED_PATIENTS
from hne.additional_signatures.paths import (ADDITIONAL_SIGNATURES, ADDITIONAL_SIGNATURES_PLOTS,
                                             NAIVE_PATIENTS_CSV, SPOT_TABLES)

CONFIG = {                            # same values as plot_fmrp_variant_maps.py
    "min_spots_per_patient": 50,
    "n_boot": 2000,
    "seed": 0,
    "surface_grid": 80,
    "surface_bandwidth": 0.07,
    "surface_min_support": 8.0,
}
DEPTH = ["total_counts", "n_genes_detected"]
# composite name -> (up set, down set), as in plot_fmrp_variant_maps.py
COMPOSITES = {"bulk": ("FMRP_bulk_up", "FMRP_bulk_down"),
              "all": ("FMRP_all_up", "FMRP_all_down"),
              "all_noAPM": ("FMRP_all_up", "FMRP_all_down_noAPM")}
APM = {"APM4": "APM4_module_score", "APM12": "APM12_module_score"}

patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
module = sorted({f"{s}_module_score" for pair in COMPOSITES.values() for s in pair} | set(APM.values()))
absent = [c for c in [*module, *DEPTH] if c not in spots.columns]
if absent:
    raise KeyError(f"spot tables lack {absent}: run add_fmrp_module_scores.py / add_spot_depth.py on the server and push the tables")
for kind, (up, down) in COMPOSITES.items():
    spots[f"FMRP_{kind}_updown_module_score"] = spots[f"{up}_module_score"] - spots[f"{down}_module_score"]
dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients")

rows = []
for score in [*module, *(f"FMRP_{kind}_updown_module_score" for kind in COMPOSITES)]:
    for depth in DEPTH:
        rho = association.per_patient_rho_pair(dominant, score, depth, CONFIG["min_spots_per_patient"])
        rows.append({"score": score.removesuffix("_score"), "depth": depth,
                     **association.patient_level_summary(rho, CONFIG["n_boot"], CONFIG["seed"])})
depth_summary = pd.DataFrame(rows)
depth_summary.to_csv(ADDITIONAL_SIGNATURES / "module_score_depth_rho_summary.csv", index=False)
print(depth_summary.round(3).to_string(index=False))

composites = {kind: (f"{up}_module_score", f"{down}_module_score") for kind, (up, down) in COMPOSITES.items()}
for suffix, covariates in (("", None), ("_depth_adjusted", DEPTH)):
    per_patient, opposition = association.plot_updown_opposition(
        dominant, composites, APM, ADDITIONAL_SIGNATURES_PLOTS / f"fmrp_updown_vs_apm_module{suffix}.png",
        CONFIG["min_spots_per_patient"], CONFIG["n_boot"], CONFIG["seed"],
        covariates=covariates, score_method="module score")
    opposition.to_csv(ADDITIONAL_SIGNATURES / f"fmrp_updown_vs_apm_module{suffix}_summary.csv", index=False)
    per_patient.to_csv(ADDITIONAL_SIGNATURES / f"fmrp_updown_vs_apm_module{suffix}_per_patient.csv", index=False)
    print(f"\nup / down vs APM, module score{suffix.replace('_', ' ')}")
    print(opposition.round(3).to_string(index=False))

for kind in COMPOSITES:
    for apm, column in APM.items():
        association.plot_surface(dominant, column, f"{apm} module",
                                 ADDITIONAL_SIGNATURES_PLOTS / f"surface_{apm}_module_by_FMRP_{kind}_module.png", CONFIG,
                                 x_score=f"FMRP_{kind}_updown_module_score",
                                 x_label=f"FMRP {kind} (up - down)")
print("plots in", ADDITIONAL_SIGNATURES_PLOTS)
