"""scripts/additional_signatures/plot_tumor_fraction_distribution.py

Step 3. Distribution of tumor fraction over all in-tissue spots of the selected naive patients, to
choose the "cancer-dominant" threshold by eye. Writes plots/tumor_fraction_distribution.png and prints the
quantiles. Then set CONFIG["cancer_dominant_threshold"] in plot_signature_associations.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hne.additional_signatures import association
from hne.additional_signatures.paths import ADDITIONAL_SIGNATURES_PLOTS, NAIVE_PATIENTS_CSV, SPOT_TABLES

# selected patients whose extraction failed, and why. The plots run on the others.
EXCLUDED_PATIENTS = {
    "CH_L_378a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_394a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_401a": "h5ad has no deconv_sample_level_custom1_weights",
    "CH_L_377a": "h5ad has no deconv_sample_level_custom1_weights",
}

CONFIG = {"n_bins": 50, "reference_lines": [0.3, 0.5, 0.7]}   # candidate thresholds drawn as guides, not decisions

patients = association.analysis_patients(NAIVE_PATIENTS_CSV, SPOT_TABLES, EXCLUDED_PATIENTS)
spots = association.load_spot_tables(SPOT_TABLES, patients)

fig, (ax_all, ax_patient) = plt.subplots(1, 2, figsize=(11, 4))
ax_all.hist(spots["tumor_fraction"], bins=CONFIG["n_bins"], color="#08519c", linewidth=0)
ax_all.set_title(f"All spots, {len(spots):,} spots of {len(patients)} patients")
for line in CONFIG["reference_lines"]:
    ax_all.axvline(line, color="#999999", linestyle="--", linewidth=1)
    ax_all.text(line, ax_all.get_ylim()[1], f" {line}", va="top", fontsize=8, color="#555555")
ax_patient.boxplot([g["tumor_fraction"].to_numpy() for _, g in spots.groupby("patient_id")],
                   tick_labels=[p.replace("CH_L_", "") for p in sorted(patients)], showfliers=False)
ax_patient.set_title("Per patient")
for ax in (ax_all, ax_patient):
    ax.set_xlabel("tumor fraction (sum of Tu_* deconvolution fractions)" if ax is ax_all else "patient")
    ax.spines[["top", "right"]].set_visible(False)
ax_all.set_ylabel("spots")
ax_patient.tick_params(axis="x", rotation=90)
fig.tight_layout()
out = ADDITIONAL_SIGNATURES_PLOTS / "tumor_fraction_distribution.png"
fig.savefig(out, dpi=200)

print(spots["tumor_fraction"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_string())
for line in CONFIG["reference_lines"]:
    print(f"share of spots with tumor_fraction >= {line}: {(spots['tumor_fraction'] >= line).mean():.2%}")
print("written:", out)
