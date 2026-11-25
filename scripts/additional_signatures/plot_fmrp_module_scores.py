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

from hne.additional_signatures import updown_report

CONFIG = {                            # same values as plot_fmrp_variant_maps.py
    "min_spots_per_patient": 50,
    "n_boot": 2000,
    "seed": 0,
    "surface_grid": 80,
    "surface_bandwidth": 0.07,
    "surface_min_support": 8.0,
}

updown_report.run(method="module", column_suffix="_module_score", score_label="module score",
                  server_script="add_fmrp_module_scores.py", depth_summary_name="module_score_depth_rho_summary.csv",
                  cfg=CONFIG)
