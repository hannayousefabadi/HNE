"""scripts/model_train/phase3/train_phase3_fm_abmil.py

Phase 3: which foundation model (FM) is the best single encoder? Phikon-v2, Virchow2 and
CONCH v1.5 patch bags go through the identical Phase 2 head (gated-attention ABMIL, distributional
mu/sigma, Gaussian beta-NLL) and the identical nested grouped CV harness. Only the input features
change. A mean-pooled ridge runs next to every ABMIL arm.

All arms are scored on exactly the same patients and tiles, so folds are identical and the
comparison is paired at the patient level.

Fixed before the first run (evaluation rules 1 and 9):
  * Hyperparameters are the Phase 2 values; nothing is tuned per FM.
  * Primary metric: within-patient r averaged over the 5 signatures, ABMIL arm.
  * Primary comparison: each of Virchow2 / CONCH v1.5 against Phikon-v2, paired patient-level
    bootstrap CI and sign-flip permutation p on the per-patient difference.
  * Secondary, reported for every signature: between-patient r, pooled r, high-vs-low AUC,
    sigma calibration, ABMIL vs ridge for each FM.
  * The FM with the highest primary metric is carried into Phase 4. Its number is selected on
    these outer folds, so it is optimistic as an estimate of generalization.
"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch

from hne.core.config import ROOT
from hne.core.paths import PATIENT_IDS, RESULTS
from hne.models.abmil import ABMIL, compute_tile_diagnostics
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.evaluation import AUC_DEFINITION, evaluate_target, paired_within_r_difference
from hne.models.mlp import GaussianBetaNLLLoss
from hne.models.ridge_cv import nested_cv_ridge
from hne.models.torch_cv import nested_cv_torch

CONFIG = {
    "output_dir": RESULTS / "phase3_fm_comparison",
    "arms": ["phikon_v2_unpooled", "virchow2_unpooled", "conch_v15_unpooled"],  # first arm is the reference
    # CONCH v1.5 files of these patients are 1024-d, written by the timm fallback loader instead of
    # TRIDENT (768-d). A different feature space, so the patients leave every arm.
    "exclude_patients": ["CH_L_275a", "CH_L_276a", "CH_L_277a", "CH_L_278a", "CH_L_279a"],
    "n_outer_folds": 5,
    "n_inner_folds": 5,
    "attention_dim": 128,
    "head_hidden_dim": 128,
    "dropout_rate": 0.25,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "epochs": 70,
    "batch_size": 32,
    "patience": 15,
    "beta": 0.5,
    "seed": 42,
    "min_tiles_within_patient": 5,
    "n_bootstrap": 2000,
    "n_permutations": 10000,
    "grad_clip_norm": 1.0,
    "top_k_attention": 5,
    "ridge_alphas": np.logspace(-1, 6, 15),   # same grid as Phase 1
}

SIGNATURES = ["FMRP", "Cell_cycle", "YAP", "WNT", "EMT"]
TARGET_COLS = [f"{sig}_signature_score" for sig in SIGNATURES]


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def evaluate_and_save(y, y_pred, groups, tile_ids, fold_of_tile, out_dir: Path, extra_pred_cols: dict | None = None):
    """Phase 1 output schema: predictions, fold assignments, per-signature metrics, per-patient r."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_frames, metric_rows, per_patient_frames = [], [], []
    for k, sig in enumerate(SIGNATURES):
        frame = pd.DataFrame({"patient_id": groups, "tile_id": tile_ids, "outer_fold": fold_of_tile,
                              "target": sig, "y_true": y[:, k], "mu_pred": y_pred[:, k]})
        for name, values in (extra_pred_cols or {}).items():
            frame[name] = values[:, k]
        pred_frames.append(frame)
        metrics, per_patient = evaluate_target(
            y_true=y[:, k].astype(np.float64), y_pred=y_pred[:, k], patient_ids=groups,
            min_tiles_within=CONFIG["min_tiles_within_patient"], n_bootstrap=CONFIG["n_bootstrap"],
            n_permutations=CONFIG["n_permutations"], seed=CONFIG["seed"],
        )
        metric_rows.append({"target": sig, **metrics})
        per_patient.insert(0, "target", sig)
        per_patient_frames.append(per_patient)

    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "cv_metrics_summary.csv", index=False)
    per_patient_df = pd.concat(per_patient_frames, ignore_index=True)
    per_patient_df.to_csv(out_dir / "cv_per_patient_metrics.csv", index=False)
    (pd.DataFrame({"patient_id": groups, "outer_fold": fold_of_tile})
       .groupby("patient_id").agg(outer_fold=("outer_fold", "first"), n_tiles=("outer_fold", "size"))
       .reset_index().to_csv(out_dir / "cv_fold_assignments.csv", index=False))
    return metrics_df, per_patient_df


def print_metrics(title, metrics_df):
    print(f"\n=== {title} ===")
    for _, m in metrics_df.iterrows():
        print(f"{m['target']:<11}| pooled r {m['pooled_r']:+.3f} | within r {m['within_patient_r']:+.3f} "
              f"[{m['within_patient_r_ci_low']:+.3f}, {m['within_patient_r_ci_high']:+.3f}] p={m['within_patient_p']:.1e} "
              f"| between r {m['between_patient_r']:+.3f} | AUC {m['high_vs_low_auc']:.3f}")


def run_arm(arm: str, restrict_to: set | None, patients: list[str], device: torch.device) -> dict:
    spec = FEATURE_REGISTRY[arm]
    arm_dir = CONFIG["output_dir"] / arm
    X, y, tile_meta = load_features_and_targets(
        features_dir=spec["dir"], patient_ids=patients, target_cols=TARGET_COLS,
        filename_suffix=spec["suffix"], as_bags=True, expected_dim=spec["dim"], restrict_to=restrict_to,
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])
    n_patients, n_tiles = len(np.unique(groups)), len(X)
    sizes = [len(b) for b in X]
    print(f"\n{'#'*100}\n# ARM {arm}: {n_patients} patients, {n_tiles} tiles, {spec['dim']}-d bags, "
          f"{np.mean(sizes):.1f} patches/tile (min {min(sizes)}, max {max(sizes)})\n{'#'*100}")

    # ---- ABMIL
    started = datetime.now().astimezone()
    cv = nested_cv_torch(
        X=X, y=y, groups=groups, tile_ids=tile_ids, target_names=SIGNATURES,
        model_factory=lambda: ABMIL(input_dim=spec["dim"], n_targets=len(SIGNATURES),
                                    attention_dim=CONFIG["attention_dim"], head_hidden_dim=CONFIG["head_hidden_dim"],
                                    dropout_rate=CONFIG["dropout_rate"], is_distributional=True),
        optimizer_factory=lambda m: torch.optim.AdamW(m.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]),
        loss_fn=GaussianBetaNLLLoss(beta=CONFIG["beta"]), is_distributional=True,
        n_outer_folds=CONFIG["n_outer_folds"], n_inner_folds=CONFIG["n_inner_folds"],
        epochs=CONFIG["epochs"], patience=CONFIG["patience"], batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"], device=device, min_tiles_within_patient=CONFIG["min_tiles_within_patient"],
        n_bootstrap=CONFIG["n_bootstrap"], n_permutations=CONFIG["n_permutations"],
        grad_clip_norm=CONFIG["grad_clip_norm"], return_attention=True,
    )
    abmil_dir = arm_dir / "abmil"
    abmil_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "metrics_summary", "per_patient_metrics", "fold_assignments", "calibration_summary"):
        file = {"predictions": "cv_predictions", "metrics_summary": "cv_metrics_summary",
                "per_patient_metrics": "cv_per_patient_metrics", "fold_assignments": "cv_fold_assignments",
                "calibration_summary": "calibration_summary"}[name]
        cv[name].to_csv(abmil_dir / f"{file}.csv", index=False)
    diag = pd.DataFrame([{"patient_id": g, "tile_id": t, **compute_tile_diagnostics(w, top_k=CONFIG["top_k_attention"])}
                         for g, t, w in zip(groups, tile_ids, cv["attention_weights"])])
    diag.to_csv(abmil_dir / "attention_diagnostics.csv", index=False)
    print(f"Selected epochs per outer fold: {cv['selected_epochs']}")
    print_metrics(f"{arm} ABMIL", cv["metrics_summary"])
    print(cv["calibration_summary"][["target", "coverage_1sigma_pct", "coverage_2sigma_pct", "error_vs_std_spearman"]].to_string(index=False))
    abmil_info = {"started": started.isoformat(timespec="seconds"),
                  "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "selected_epochs_per_outer_fold": cv["selected_epochs"]}
    abmil_metrics, abmil_per_patient, calibration = cv["metrics_summary"], cv["per_patient_metrics"], cv["calibration_summary"]

    # ---- ridge on the mean of each bag, same tiles and same folds
    X_mean = np.stack([b.mean(axis=0) for b in X])
    del X
    y_pred, fold_of_tile, alpha_records = nested_cv_ridge(
        X_mean, y, groups, alphas=CONFIG["ridge_alphas"], n_outer_folds=CONFIG["n_outer_folds"],
        n_inner_folds=CONFIG["n_inner_folds"], seed=CONFIG["seed"],
    )
    for record in alpha_records:
        record["target"] = SIGNATURES[record.pop("target_index")]
    ridge_dir = arm_dir / "ridge"
    ridge_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alpha_records).to_csv(ridge_dir / "selected_alphas.csv", index=False)
    ridge_metrics, ridge_per_patient = evaluate_and_save(y, y_pred, groups, tile_ids, fold_of_tile, ridge_dir)
    print_metrics(f"{arm} mean-pooled ridge", ridge_metrics)
    edge = pd.DataFrame(alpha_records)["at_grid_edge"].sum()
    if edge:
        print(f"WARNING: {edge} ridge alpha selections sit on the grid edge")

    return {"abmil_metrics": abmil_metrics, "abmil_per_patient": abmil_per_patient, "calibration": calibration,
            "ridge_metrics": ridge_metrics, "ridge_per_patient": ridge_per_patient, "diag": diag,
            "n_patients": int(n_patients), "n_tiles": int(n_tiles), "abmil_info": abmil_info,
            "tile_keys": set(zip(groups.tolist(), tile_ids.tolist()))}


def run():
    started_at = datetime.now().astimezone()
    out = CONFIG["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPhase 3 FM comparison | Device: {device} | arms: {CONFIG['arms']}")

    # === common cohort: patients and tiles that every arm has a bag for
    patients = None
    for arm in CONFIG["arms"]:
        ids = set(get_available_patient_ids(PATIENT_IDS, FEATURE_REGISTRY[arm])) - set(CONFIG["exclude_patients"])
        patients = ids if patients is None else patients & ids
    patients = sorted(patients)
    common_tiles = None
    for arm in CONFIG["arms"]:
        spec = FEATURE_REGISTRY[arm]
        keys = set()
        for pid in patients:
            for f in Path(spec["dir"]).glob(f"{pid}_*_{spec['suffix']}.npy"):
                keys.add((pid, f.name[len(pid) + 1: -len(f"_{spec['suffix']}.npy")]))
        common_tiles = keys if common_tiles is None else common_tiles & keys
    print(f"Common cohort: {len(patients)} patients, {len(common_tiles)} candidate tiles "
          f"(tiles without a target score are dropped by the loader)")

    results = {}
    for arm in CONFIG["arms"]:
        results[arm] = run_arm(arm, common_tiles, patients, device)
    sets = {tuple(sorted(r["tile_keys"])) for r in results.values()}
    assert len(sets) == 1, "arms were scored on different tiles"

    # === comparison tables
    ref = CONFIG["arms"][0]
    summary_rows = []
    for arm, r in results.items():
        for kind in ("abmil", "ridge"):
            for _, m in r[f"{kind}_metrics"].iterrows():
                summary_rows.append({"arm": arm, "head": kind, **m.to_dict()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "phase3_metrics_all_arms.csv", index=False)

    paired = []
    for arm in CONFIG["arms"]:
        if arm != ref:
            df = paired_within_r_difference(results[arm]["abmil_per_patient"], results[ref]["abmil_per_patient"],
                                            CONFIG["n_bootstrap"], CONFIG["n_permutations"], CONFIG["seed"])
            df.insert(0, "comparison", f"{arm} ABMIL - {ref} ABMIL")
            paired.append(df)
    for arm in CONFIG["arms"]:
        df = paired_within_r_difference(results[arm]["abmil_per_patient"], results[arm]["ridge_per_patient"],
                                        CONFIG["n_bootstrap"], CONFIG["n_permutations"], CONFIG["seed"])
        df.insert(0, "comparison", f"{arm} ABMIL - {arm} ridge")
        paired.append(df)
    paired_df = pd.concat(paired, ignore_index=True)
    paired_df.to_csv(out / "phase3_paired_within_r_differences.csv", index=False)

    ranking = (summary[summary["head"] == "abmil"].groupby("arm")["within_patient_r"].mean()
               .sort_values(ascending=False).rename("mean_within_patient_r_5_signatures"))
    ranking.to_csv(out / "phase3_ranking.csv")

    print(f"\n{'='*100}\n=== PHASE 3: mean within-patient r over 5 signatures (ABMIL) ===")
    print(ranking.to_string())
    print(f"\n=== Paired patient-level differences in within-patient r ===")
    print(paired_df[["comparison", "target", "mean_diff_a_minus_b", "diff_ci_low", "diff_ci_high", "p_value"]].to_string(index=False))

    with open(out / "run_summary.json", "w") as f:
        json.dump({
            "analysis_started_at": started_at.isoformat(timespec="seconds"),
            "analysis_finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git": git_state(), "phase": "phase3_fm_comparison", "arms": CONFIG["arms"],
            "n_patients": len(patients), "n_tiles": results[ref]["n_tiles"],
            "excluded_patients": CONFIG["exclude_patients"],
            "abmil_runs": {arm: r["abmil_info"] for arm, r in results.items()},
            "ranking_primary_metric": ranking.to_dict(),
            "config": {k: (v.tolist() if isinstance(v, np.ndarray) else str(v) if isinstance(v, Path) else v)
                       for k, v in CONFIG.items()},
            "auc_definition": AUC_DEFINITION,
        }, f, indent=2)
    print(f"\nAll outputs written to: {out}")


if __name__ == "__main__":
    run()
