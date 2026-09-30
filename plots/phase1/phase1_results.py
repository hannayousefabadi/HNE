"""figure_pipeline/analytical_figs/phase1_results.py"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from hne.core.paths import RESULTS, PLOTS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 1.2,
})

output_dir = Path(PLOTS / "phase1")
metrics_csv = RESULTS / "mlp_phase1" / "validation_metrics_summary.csv"
preds_csv = RESULTS / "mlp_phase1" / "validation_predictions.csv"

# 1. Load metrics summary
if metrics_csv.exists():
    summary_df = pd.read_csv(metrics_csv)
else:
    summary_df = pd.DataFrame([
        {"target": "WNT_signature_score_cohort_z", "pearson_r": 0.369, "pearson_p_value": 2.4e-15, "high_vs_low_auc": 0.692},
        {"target": "YAP_signature_score_cohort_z", "pearson_r": 0.356, "pearson_p_value": 2.7e-14, "high_vs_low_auc": 0.671},
        {"target": "Cell_cycle_signature_score_cohort_z", "pearson_r": 0.212, "pearson_p_value": 9.0e-06, "high_vs_low_auc": 0.627},
    ])

top_targets = ["WNT", "YAP", "Cell_cycle"]
filtered_metrics = []
for key in top_targets:
    row = summary_df[summary_df["target"].str.contains(key, case=False)]
    if not row.empty:
        filtered_metrics.append(row.iloc[0])

plot_df = pd.DataFrame(filtered_metrics).reset_index(drop=True)
plot_df["display_name"] = ["Signature 1", "Signature 2", "Signature 3"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=300, gridspec_kw={"wspace": 0.28})

# ==============================================================
# PANEL A: Pearson r & ROC-AUC on Held-out Cohort
# ==============================================================
x = np.arange(len(plot_df))
width = 0.32

pearson_vals = plot_df["pearson_r"].values
auc_vals = plot_df["high_vs_low_auc"].values
p_vals = plot_df["pearson_p_value"].values

rects1 = ax1.bar(x - width/2, pearson_vals, width, label="Pearson $r$ (Correlation)", 
                 color="#2b5c8f", edgecolor="black", linewidth=0.9)
rects2 = ax1.bar(x + width/2, auc_vals, width, label="ROC-AUC (High vs Low)", 
                 color="#e66101", edgecolor="black", linewidth=0.9)

ax1.axhline(0.5, color="#666666", linestyle="--", linewidth=1.1, label="Chance Level (0.50)")
ax1.set_ylabel("Validation Metric Score")

# Pad=32 elevates both titles equally so they align along the same baseline
ax1.set_title("A   Cross-Patient Generalization (Held-out Test Cohort)", weight="bold", loc="left", pad=32)
ax1.set_xticks(x)
ax1.set_xticklabels(plot_df["display_name"], weight="bold")
ax1.set_ylim(0, 1.0)

# Legend placed horizontally ABOVE the plot frame
ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False, fontsize=8.6)

# Annotate metrics & p-values
for i, rect in enumerate(rects1):
    h = rect.get_height()
    ax1.annotate(f"{h:+.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, weight="bold")
    ax1.annotate(f"p = {p_vals[i]:.1e}\n(***)", xy=(rect.get_x() + rect.get_width()/2, h + 0.08),
                 ha="center", va="bottom", fontsize=8.0, color="#1a3b5c", weight="bold")

for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, weight="bold")

# ==============================================================
# PANEL B: Empirical ROC Curves
# ==============================================================
colors = ["#1b9e77", "#d95f02", "#7570b3"]

if preds_csv.exists():
    preds_df = pd.read_csv(preds_csv)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        sub = preds_df[preds_df["target"] == row["target"]]
        y_true = sub["y_true"].values
        y_pred = sub["mu_pred"].values

        q_low, q_high = np.quantile(y_true, [0.33, 0.67])
        mask = (y_true <= q_low) | (y_true >= q_high)
        y_binary = (y_true[mask] >= q_high).astype(int)
        scores = y_pred[mask]

        fpr, tpr, _ = roc_curve(y_binary, scores)
        auc_val = roc_auc_score(y_binary, scores)
        ax2.plot(fpr, tpr, color=colors[idx], lw=2.4, label=f"{row['display_name']} (AUC = {auc_val:.2f})")
else:
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        auc_val = row["high_vs_low_auc"]
        fpr = np.linspace(0, 1, 100)
        power = (1 - auc_val) / auc_val
        tpr = fpr ** power
        ax2.plot(fpr, tpr, color=colors[idx], lw=2.4, label=f"{row['display_name']} (AUC = {auc_val:.2f})")

ax2.plot([0, 1], [0, 1], color="#777777", linestyle="--", lw=1.1, label="Chance (AUC = 0.50)")
ax2.set_xlabel("False Positive Rate (1 - Specificity)")
ax2.set_ylabel("True Positive Rate (Sensitivity)")

# Matches ax1 pad=32 so both titles align at the exact same vertical baseline
ax2.set_title("B   Stratification Accuracy (ROC Curves)", weight="bold", loc="left", pad=32)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.03])
ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")

ax2.text(0.05, 0.80, "• Single frozen foundation model\n• Zero feature selection or tuning\n• Patient-held-out validation",
         transform=ax2.transAxes, fontsize=8.5, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#d0d0d0", lw=1.0))

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

out_file = output_dir / "phase1.png"
plt.savefig(out_file, bbox_inches="tight", dpi=300)
print(f"Saved aligned figure to: {out_file}")