"""scripts/audit/purity_control_by_group.py

Registration audit, part C (local, after the cluster results are pulled): rerun only the
tumor-purity positive control, per group of patients, through the Phase 1 harness
(same nested grouped CV, alpha grid, seed and metrics; nothing about the harness changes).

Groups:
  resolution   from slide_inventory.csv: "scan" (effective um/px <= scan_max_um_per_px) or "downscaled"
  verdict      from overlay_verdicts.csv, filled in by hand after viewing the overlays:
               patient_id,verdict   with verdict = aligned | misaligned
If overlay_verdicts.csv does not exist, an empty template is written and only the
resolution groups are run.
"""
from datetime import datetime
import json
import numpy as np
import pandas as pd

from hne.core.paths import PATIENT_IDS, REGISTRATION_AUDIT
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.evaluation import evaluate_target
from hne.models.ridge_cv import nested_cv_ridge

# CV settings are the Phase 1 ones (scripts/model_train/phase1/train_phase1_ridge.py); do not change them here
CONFIG = {
    "model_name": "phikon_v2",
    "inventory_csv": REGISTRATION_AUDIT / "slide_inventory.csv",
    "verdicts_csv": REGISTRATION_AUDIT / "overlay_verdicts.csv",
    "output_dir": REGISTRATION_AUDIT,
    "control_target": "tile_purity",
    "scan_max_um_per_px": 0.6,
    "min_patients_per_group": 15,   # 5 outer x 5 inner folds need enough patients in every fold
    "n_outer_folds": 5,
    "n_inner_folds": 5,
    "alphas": np.logspace(-1, 6, 15),
    "seed": 42,
    "min_tiles_within_patient": 5,
    "n_bootstrap": 2000,
    "n_permutations": 10000,
}


def load_groups() -> pd.DataFrame:
    if not CONFIG["inventory_csv"].exists():
        raise FileNotFoundError(f"{CONFIG['inventory_csv']} not found. Run scripts/audit/registration_audit_cluster.py "
                                f"on the cluster and pull its results first.")
    inv = pd.read_csv(CONFIG["inventory_csv"])
    groups = inv[["patient_id", "effective_um_per_px"]].copy()
    groups["resolution"] = np.where(groups["effective_um_per_px"] <= CONFIG["scan_max_um_per_px"], "scan", "downscaled")

    if CONFIG["verdicts_csv"].exists():
        verdicts = pd.read_csv(CONFIG["verdicts_csv"])
        verdicts["verdict"] = verdicts["verdict"].fillna("").astype(str).str.strip().str.lower()
        unknown = set(verdicts["verdict"]) - {"aligned", "misaligned", ""}
        if unknown:
            raise ValueError(f"Unknown verdicts in {CONFIG['verdicts_csv']}: {sorted(unknown)}. Use aligned or misaligned.")
        groups = groups.merge(verdicts[["patient_id", "verdict"]], on="patient_id", how="left")
        groups["verdict"] = groups["verdict"].replace("", np.nan).fillna("unreviewed")
    else:
        groups[["patient_id"]].assign(verdict="").to_csv(CONFIG["verdicts_csv"], index=False)
        print(f"No overlay verdicts yet. Wrote a template to {CONFIG['verdicts_csv']}; "
              f"fill in aligned / misaligned and rerun. Running the resolution groups only.")
        groups["verdict"] = "unreviewed"
    return groups


def run():
    output_dir = CONFIG["output_dir"]
    groups_df = load_groups()
    spec = FEATURE_REGISTRY[CONFIG["model_name"]]

    X, y, tile_meta = load_features_and_targets(
        features_dir=spec["dir"], patient_ids=get_available_patient_ids(PATIENT_IDS, spec),
        target_cols=[CONFIG["control_target"]], filename_suffix=spec["suffix"],
    )
    patient_of_tile = np.array([pid for pid, _ in tile_meta])

    # one selection of patients per group; "all" is the reference
    selections = {("all", "all"): set(groups_df["patient_id"])}
    for column in ("resolution", "verdict"):
        for value, sub in groups_df.groupby(column):
            selections[(column, value)] = set(sub["patient_id"])
    for (res, ver), sub in groups_df.groupby(["resolution", "verdict"]):
        selections[("resolution+verdict", f"{res}+{ver}")] = set(sub["patient_id"])

    rows = []
    for (grouping, value), patients in selections.items():
        mask = np.isin(patient_of_tile, list(patients))
        n_patients = len(np.unique(patient_of_tile[mask]))
        row = {"grouping": grouping, "group": value, "n_patients_in_group": len(patients),
               "n_patients_with_features": n_patients, "n_tiles": int(mask.sum())}
        if n_patients < CONFIG["min_patients_per_group"]:
            row["status"] = f"skipped: fewer than {CONFIG['min_patients_per_group']} patients with features"
            rows.append(row)
            print(f"{grouping}={value}: {n_patients} patients, skipped")
            continue

        y_pred, _, fold_records = nested_cv_ridge(
            X[mask], y[mask], patient_of_tile[mask], alphas=CONFIG["alphas"],
            n_outer_folds=CONFIG["n_outer_folds"], n_inner_folds=CONFIG["n_inner_folds"], seed=CONFIG["seed"],
        )
        metrics, _ = evaluate_target(
            y_true=y[mask][:, 0].astype(np.float64), y_pred=y_pred[:, 0], patient_ids=patient_of_tile[mask],
            min_tiles_within=CONFIG["min_tiles_within_patient"], n_bootstrap=CONFIG["n_bootstrap"],
            n_permutations=CONFIG["n_permutations"], seed=CONFIG["seed"],
        )
        row.update({"status": "ok", **metrics,
                    "n_folds_alpha_at_grid_max": sum(r["selected_alpha"] == CONFIG["alphas"][-1] for r in fold_records)})
        rows.append(row)
        print(f"{grouping}={value}: {n_patients} patients, {int(mask.sum())} tiles | within-patient r "
              f"{metrics['within_patient_r']:+.3f} [{metrics['within_patient_r_ci_low']:+.3f}, "
              f"{metrics['within_patient_r_ci_high']:+.3f}] p={metrics['within_patient_p']:.1e} | "
              f"between-patient r {metrics['between_patient_r']:+.3f}")

    pd.DataFrame(rows).to_csv(output_dir / "purity_control_by_group.csv", index=False)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(output_dir / "purity_control_run.json", "w") as f:
        json.dump({"analysis_finished_at": finished_at, "control_target": CONFIG["control_target"],
                   "features": CONFIG["model_name"],
                   "config": {k: (v.tolist() if isinstance(v, np.ndarray) else str(v)) for k, v in CONFIG.items()}}, f, indent=2)
    print(f"\nFinished at {finished_at}. Written to {output_dir / 'purity_control_by_group.csv'}")
    print("Read within-patient r: it is not affected by the cross-validation fold-offset artifact.")


if __name__ == "__main__":
    run()
