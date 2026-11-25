"""scripts/additional_signatures/plot_fmrp_singscores.py

Runs locally from the pushed spot tables (after add_fmrp_singscores.py and add_spot_depth.py have been run on the
server). The FMRP up / down analysis redone with singscore (Foroutan 2018, the method built for up / down signatures;
hne.additional_signatures.singscore) for every set. Composite per signature = singscore TotalScore =
score(up set) - score(down set). The ssGSEA and module-score outputs are not touched; every file here has
"singscore" in its name (see hne.additional_signatures.updown_report):
  plots/surface_{APM4,APM12}_singscore_by_FMRP_{bulk,all,all_noAPM}_singscore.png
  plots/fmrp_updown_vs_apm_singscore.png, plots/fmrp_updown_vs_apm_singscore_depth_adjusted.png (+ their CSVs)
  singscore_depth_rho_summary.csv
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

updown_report.run(method="singscore", column_suffix="_singscore", score_label="singscore",
                  server_script="add_fmrp_singscores.py", depth_summary_name="singscore_depth_rho_summary.csv",
                  cfg=CONFIG)
