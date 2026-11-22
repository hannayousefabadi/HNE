"""scripts/additional_signatures/plot_signature_associations.py

Step 4. Keeps the cancer-dominant spots of the selected naive patients and plots FMRP, APM4 and
APM12 against the immune count within 200 um.

  plots/surface_APM4.png, plots/surface_APM12.png   reference-style: colour = APM score over FMRP x immune count
  plots/surface_APM4_FMR1.png, plots/surface_APM12_FMR1.png   the same with FMR1 expression on the x axis
  plots/trend_vs_immune_count.png                   FMRP, APM4, APM12 each against immune count, patient-bootstrap band
  patient_level_summary.csv                         mean per-patient Spearman rho, 95% bootstrap CI, permutation p

The threshold is read off the tumor-fraction distribution from step 3 and set below by hand.
Do not change it after looking at these plots to improve a result.
"""
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.paths import (ADDITIONAL_SIGNATURES, ADDITIONAL_SIGNATURES_PLOTS,
                                             NAIVE_PATIENTS_CSV, SPOT_TABLES)

# selected patients whose extraction failed, and why. The plots run on the others.
EXCLUDED_PATIENTS = {
    "CH_L_378a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_394a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_401a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_377a": "h5ad has no deconv_sample_level_custom1_weights",
}

CONFIG = {
    "cancer_dominant_threshold": 0.5,    # tumor_fraction >= this keeps a spot: tumor outweighs all other cell types together
    "min_spots_per_patient": 50,         # a patient with fewer cancer-dominant spots is left out of the statistics
    "n_bins": 8,                         # immune-count quantile bins of the trend plot
    "n_boot": 2000,
    "seed": 0,
    "surface_grid": 80,
    "surface_bandwidth": 0.07,           # kernel sd as a fraction of the axis range
    "surface_min_support": 8.0,          # kernel mass (in spots) a grid cell needs to be drawn
}
SCORES = ["FMRP_score", "FMR1_expression", "APM4_score", "APM12_score"]

if CONFIG["cancer_dominant_threshold"] is None:
    raise RuntimeError("set CONFIG['cancer_dominant_threshold'] after looking at plot_tumor_fraction_distribution.py")

patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)
dominant = spots[spots["tumor_fraction"] >= CONFIG["cancer_dominant_threshold"]]
per_patient = dominant.groupby("patient_id").size()
kept = per_patient[per_patient >= CONFIG["min_spots_per_patient"]].index
print(f"{len(dominant):,}/{len(spots):,} spots cancer-dominant; {len(kept)}/{len(patients)} patients with "
      f">= {CONFIG['min_spots_per_patient']} such spots")
dominant = dominant[dominant["patient_id"].isin(kept)]

summaries, trends, rows = {}, {}, []
for score in SCORES:
    rho = association.per_patient_rho(dominant, score, CONFIG["min_spots_per_patient"])
    summaries[score.removesuffix("_score").removesuffix("_expression")] = association.patient_level_summary(rho, CONFIG["n_boot"], CONFIG["seed"])
    trends[score.removesuffix("_score").removesuffix("_expression")] = association.binned_trend(dominant, score, CONFIG["n_bins"],
                                                                    CONFIG["n_boot"], CONFIG["seed"])
    rows.append({"score": score.removesuffix("_score").removesuffix("_expression"), **summaries[score.removesuffix("_score").removesuffix("_expression")]})
summary = pd.DataFrame(rows)
summary.to_csv(ADDITIONAL_SIGNATURES / "patient_level_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

for apm in ("APM4", "APM12"):
    association.plot_surface(dominant, f"{apm}_score", f"{apm} score",
                             ADDITIONAL_SIGNATURES_PLOTS / f"surface_{apm}.png", CONFIG)
    association.plot_surface(dominant, f"{apm}_score", f"{apm} score",
                             ADDITIONAL_SIGNATURES_PLOTS / f"surface_{apm}_FMR1.png", CONFIG,
                             x_score="FMR1_expression", x_label="FMR1")
association.plot_trends(trends, summaries, ADDITIONAL_SIGNATURES_PLOTS / "trend_vs_immune_count.png")
print("plots in", ADDITIONAL_SIGNATURES_PLOTS)
