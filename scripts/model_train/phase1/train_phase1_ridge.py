"""scripts/model_train/phase1/train_phase1_ridge.py

Clean Phase 1 baseline: ridge regression on mean-pooled tile embeddings, evaluated with
nested grouped (by patient) cross-validation over the whole cohort.
"""
from datetime import datetime
import json
import subprocess
import numpy as np
import pandas as pd

from hne.core.config import ROOT
from hne.core.paths import PATIENT_IDS, RESULTS
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.evaluation import AUC_DEFINITION, evaluate_target
from hne.models.ridge_cv import nested_cv_ridge

# Everything below was fixed before the first run. Changing a value after looking at the
# outer-fold results turns those results into a selection, not a report.
CONFIG = {
    "model_name": "phikon_v2",
    "output_dir": RESULTS / "phase1_ridge",
    "n_outer_folds": 5,
    "n_inner_folds": 5,
    "alphas": np.logspace(-1, 6, 15),   # half-decade steps, 0.1 to 1e6
    "seed": 42,
    "min_tiles_within_patient": 5,      # patients with fewer tiles get no within-patient r
    "n_bootstrap": 2000,
    "n_permutations": 10000,
}

SIGNATURES = ["FMRP", "Cell_cycle", "YAP", "WNT", "EMT"]
TARGET_COLS = [f"{sig}_signature_score" for sig in SIGNATURES]


# ==== Helpers ====
def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def run():
    started_at = datetime.now().astimezone()
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_spec = FEATURE_REGISTRY[CONFIG["model_name"]]

    # === cohort discovery and loading (raw signature scores; z-scoring happens per fold)
    patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    X, y, tile_meta = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=patients,
        target_cols=TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])
    print(f"\nCohort: {len(np.unique(groups))} patients, {len(X)} tiles, {X.shape[1]}-d {CONFIG['model_name']} features")
    print(f"Nested grouped CV: {CONFIG['n_outer_folds']} outer x {CONFIG['n_inner_folds']} inner folds, "
          f"{len(CONFIG['alphas'])} alphas, seed {CONFIG['seed']}")

    # === nested grouped CV: every patient is in exactly one outer test fold
    y_pred, fold_of_tile, alpha_records = nested_cv_ridge(
        X, y, groups,
        alphas=CONFIG["alphas"],
        n_outer_folds=CONFIG["n_outer_folds"],
        n_inner_folds=CONFIG["n_inner_folds"],
        seed=CONFIG["seed"],
    )
    for record in alpha_records:
        record["target"] = SIGNATURES[record.pop("target_index")]

    for fold in range(CONFIG["n_outer_folds"]):
        chosen = [r for r in alpha_records if r["outer_fold"] == fold]
        print(f"Outer fold {fold}: {chosen[0]['n_train_patients']} train / {chosen[0]['n_test_patients']} test patients | "
              f"alphas: " + ", ".join(f"{r['target']}={r['selected_alpha']:.3g}" for r in chosen))

    # === persist predictions, folds and selected alphas
    predictions_df = pd.concat([
        pd.DataFrame({
            "patient_id": groups,
            "tile_id": tile_ids,
            "outer_fold": fold_of_tile,
            "target": sig,
            "y_true": y[:, k],
            "mu_pred": y_pred[:, k],
        }) for k, sig in enumerate(SIGNATURES)
    ], ignore_index=True)
    predictions_df.to_csv(output_dir / "cv_predictions.csv", index=False)

    alphas_df = pd.DataFrame(alpha_records)
    alphas_df.to_csv(output_dir / "selected_alphas.csv", index=False)

    (pd.DataFrame({"patient_id": groups, "outer_fold": fold_of_tile})
       .groupby("patient_id").agg(outer_fold=("outer_fold", "first"), n_tiles=("outer_fold", "size"))
       .reset_index().to_csv(output_dir / "cv_fold_assignments.csv", index=False))

    # === patient-level evaluation, all signatures
    metric_rows, per_patient_frames = [], []
    for k, sig in enumerate(SIGNATURES):
        metrics, per_patient = evaluate_target(
            y_true=y[:, k].astype(np.float64),
            y_pred=y_pred[:, k],
            patient_ids=groups,
            min_tiles_within=CONFIG["min_tiles_within_patient"],
            n_bootstrap=CONFIG["n_bootstrap"],
            n_permutations=CONFIG["n_permutations"],
            seed=CONFIG["seed"],
        )
        metric_rows.append({"target": sig, **metrics})
        per_patient.insert(0, "target", sig)
        per_patient_frames.append(per_patient)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    pd.concat(per_patient_frames, ignore_index=True).to_csv(output_dir / "cv_per_patient_metrics.csv", index=False)

    print(f"\n{'='*118}")
    print(f"=== OUTER-FOLD TEST RESULTS ({len(X)} tiles, {len(np.unique(groups))} patients) | 95% CI: patient-level bootstrap ===")
    print(f"{'='*118}")
    for _, m in metrics_df.iterrows():
        print(f"{m['target']:<11}| pooled r {m['pooled_r']:+.3f} [{m['pooled_r_ci_low']:+.3f}, {m['pooled_r_ci_high']:+.3f}] "
              f"| within r {m['within_patient_r']:+.3f} [{m['within_patient_r_ci_low']:+.3f}, {m['within_patient_r_ci_high']:+.3f}] p={m['within_patient_p']:.1e} "
              f"| between r {m['between_patient_r']:+.3f} [{m['between_patient_r_ci_low']:+.3f}, {m['between_patient_r_ci_high']:+.3f}] p={m['between_patient_p']:.1e} "
              f"| AUC {m['high_vs_low_auc']:.3f}")
    print(f"{'='*118}")
    print(f"AUC: {AUC_DEFINITION}\n")

    # === run summary
    finished_at = datetime.now().astimezone()
    summary_info = {
        "analysis_finished_at": finished_at.isoformat(timespec="seconds"),
        "analysis_started_at": started_at.isoformat(timespec="seconds"),
        "git": git_state(),
        "model": "ridge",
        "features": CONFIG["model_name"],
        "targets": SIGNATURES,
        "n_patients": int(len(np.unique(groups))),
        "n_tiles": int(len(X)),
        "config": {key: (value.tolist() if isinstance(value, np.ndarray) else str(value) if key == "output_dir" else value)
                   for key, value in CONFIG.items()},
        "alpha_selection": "per signature, lowest inner-CV mean squared error on z-scored targets",
        "scaling": "features standardized and targets z-scored with training-fold statistics only",
        "prediction_units": "raw tile signature score (back-transformed from the fold's z-scale)",
        "ci_method": "percentile 95% CI, patient-level bootstrap",
        "p_value_method": {
            "within_patient_p": "two-sided sign-flip permutation over per-patient r",
            "between_patient_p": "two-sided permutation of patient mean predictions across patients",
            "pooled_r": "no p-value: tiles are not independent; see the bootstrap CI",
        },
        "auc_definition": AUC_DEFINITION,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary_info, f, indent=2)

    print(f"Finished at {summary_info['analysis_finished_at']}. Outputs written to: {output_dir}")


if __name__ == "__main__":
    run()
