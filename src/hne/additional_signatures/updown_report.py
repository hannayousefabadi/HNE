"""src/hne/additional_signatures/updown_report.py

The FMRP up / down vs APM report for one alternative scoring method (module score, singscore), shared by
plot_fmrp_module_scores.py and plot_fmrp_singscores.py. Every set (FMRP variants, APM4, APM12) is read from the
column "{set}{column_suffix}" of the spot tables; composite per signature = score(up set) - score(down set).
Files are named after `method`, so the ssGSEA outputs and the other methods' outputs are never overwritten:
  plots/surface_{APM}_{method}_by_FMRP_{kind}_{method}.png
  plots/fmrp_updown_vs_apm_{method}.png, plots/fmrp_updown_vs_apm_{method}_depth_adjusted.png (+ _summary.csv,
  _per_patient.csv) and `depth_summary_name` (each score against each spot-depth column).
"""
import pandas as pd

from hne.additional_signatures import association, paths
from hne.additional_signatures.analysis_config import CANCER_DOMINANT_THRESHOLD, EXCLUDED_PATIENTS

DEPTH = ["total_counts", "n_genes_detected"]
# composite name -> (up set, down set), as in plot_fmrp_variant_maps.py
COMPOSITES = {"bulk": ("FMRP_bulk_up", "FMRP_bulk_down"),
              "all": ("FMRP_all_up", "FMRP_all_down"),
              "all_noAPM": ("FMRP_all_up", "FMRP_all_down_noAPM")}


def run(method: str, column_suffix: str, score_label: str, server_script: str, depth_summary_name: str, cfg: dict):
    apm = {name: f"{name}{column_suffix}" for name in ("APM4", "APM12")}
    patients = association.analysis_patients(paths.NAIVE_PATIENTS_CSV, paths.SPOT_TABLES, EXCLUDED_PATIENTS)
    spots = association.load_spot_tables(paths.SPOT_TABLES, patients)
    scores = sorted({f"{s}{column_suffix}" for pair in COMPOSITES.values() for s in pair} | set(apm.values()))
    absent = [c for c in [*scores, *DEPTH] if c not in spots.columns]
    if absent:
        raise KeyError(f"spot tables lack {absent}: run {server_script} / add_spot_depth.py on the server and push the tables")
    for kind, (up, down) in COMPOSITES.items():
        spots[f"FMRP_{kind}_updown{column_suffix}"] = spots[f"{up}{column_suffix}"] - spots[f"{down}{column_suffix}"]
    dominant = spots[spots["tumor_fraction"] >= CANCER_DOMINANT_THRESHOLD]
    print(f"{len(dominant):,} cancer-dominant spots of {len(patients)} patients")

    rows = []
    for score in [*scores, *(f"FMRP_{kind}_updown{column_suffix}" for kind in COMPOSITES)]:
        for depth in DEPTH:
            rho = association.per_patient_rho_pair(dominant, score, depth, cfg["min_spots_per_patient"])
            rows.append({"score": score.removesuffix("_score"), "depth": depth,
                         **association.patient_level_summary(rho, cfg["n_boot"], cfg["seed"])})
    depth_summary = pd.DataFrame(rows)
    depth_summary.to_csv(paths.ADDITIONAL_SIGNATURES / depth_summary_name, index=False)
    print(depth_summary.round(3).to_string(index=False))

    composites = {kind: (f"{up}{column_suffix}", f"{down}{column_suffix}") for kind, (up, down) in COMPOSITES.items()}
    for suffix, covariates in (("", None), ("_depth_adjusted", DEPTH)):
        per_patient, opposition = association.plot_updown_opposition(
            dominant, composites, apm, paths.ADDITIONAL_SIGNATURES_PLOTS / f"fmrp_updown_vs_apm_{method}{suffix}.png",
            cfg["min_spots_per_patient"], cfg["n_boot"], cfg["seed"], covariates=covariates, score_method=score_label)
        opposition.to_csv(paths.ADDITIONAL_SIGNATURES / f"fmrp_updown_vs_apm_{method}{suffix}_summary.csv", index=False)
        per_patient.to_csv(paths.ADDITIONAL_SIGNATURES / f"fmrp_updown_vs_apm_{method}{suffix}_per_patient.csv", index=False)
        print(f"\nup / down vs APM, {score_label}{suffix.replace('_', ' ')}")
        print(opposition.round(3).to_string(index=False))

    for kind in COMPOSITES:
        for name, column in apm.items():
            association.plot_surface(dominant, column, f"{name} {method}",
                                     paths.ADDITIONAL_SIGNATURES_PLOTS / f"surface_{name}_{method}_by_FMRP_{kind}_{method}.png",
                                     cfg, x_score=f"FMRP_{kind}_updown{column_suffix}", x_label=f"FMRP {kind} (up - down)",
                                     max_xticks=6)
    print("plots in", paths.ADDITIONAL_SIGNATURES_PLOTS)
