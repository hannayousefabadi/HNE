"""scripts/model_train/phase2/train_phase2_abmil.py

Phase 2: Attention-Based Multiple Instance Learning (ABMIL) on tile patch bags.
Learns patch attention weights to detect localized pathway hotspots while outputting
spatial uncertainty (sigma) and attention entropy readouts of intratumoral heterogeneity.

Evaluated with nested grouped (by patient) cross-validation and patient-level statistics.
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
from hne.models.evaluation import AUC_DEFINITION
from hne.models.mlp import GaussianBetaNLLLoss
from hne.models.torch_cv import nested_cv_torch

CONFIG = {
    "model_name": "phikon_v2_unpooled",
    "output_dir": RESULTS / "phase2_abmil",
    "n_outer_folds": 5,
    "n_inner_folds": 5,
    "input_dim": 1024,
    "attention_dim": 128,              # L = 128
    "head_hidden_dim": 128,
    "dropout_rate": 0.25,              # 0.25 on attention weights and trunk
    "lr": 5e-4,                        # AdamW lr = 5e-4
    "weight_decay": 1e-3,              # weight_decay = 1e-3
    "epochs": 70,
    "batch_size": 32,                  # 16-32 tiles
    "patience": 15,
    "beta": 0.5,                       # beta-weighting for stabilized Gaussian NLL
    "seed": 42,
    "min_tiles_within_patient": 5,     # patients with fewer tiles get no within-patient r
    "n_bootstrap": 2000,
    "n_permutations": 10000,
    "grad_clip_norm": 1.0,
    "top_k_attention": 5,              # Top-k patches for attention dominance ratio
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


def run():
    started_at = datetime.now().astimezone()
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_spec = FEATURE_REGISTRY[CONFIG["model_name"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPhase 2 ABMIL Training | Device: {device}")

    # === cohort discovery and loading
    patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    X, y, tile_meta = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=patients,
        target_cols=TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
        as_bags=True,
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])
    n_targets = len(TARGET_COLS)

    n_patients = len(np.unique(groups))
    n_tiles = len(X)
    patch_counts = [len(bag) for bag in X]
    print(f"Cohort: {n_patients} patients, {n_tiles} tiles, {CONFIG['input_dim']}-d unpooled patch bags")
    print(f"Bag sizes: mean {np.mean(patch_counts):.1f} patches/tile (min {np.min(patch_counts)}, max {np.max(patch_counts)})")
    print(f"Nested grouped CV: {CONFIG['n_outer_folds']} outer x {CONFIG['n_inner_folds']} inner folds, seed {CONFIG['seed']}")

    # === Model and optimizer factories
    def model_factory():
        return ABMIL(
            input_dim=CONFIG["input_dim"],
            n_targets=n_targets,
            attention_dim=CONFIG["attention_dim"],
            head_hidden_dim=CONFIG["head_hidden_dim"],
            dropout_rate=CONFIG["dropout_rate"],
            is_distributional=True,
        )

    def optimizer_factory(model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

    loss_fn = GaussianBetaNLLLoss(beta=CONFIG["beta"])

    print("\nStarting nested grouped cross-validation via torch_cv...")
    cv_results = nested_cv_torch(
        X=X,
        y=y,
        groups=groups,
        tile_ids=tile_ids,
        target_names=SIGNATURES,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        loss_fn=loss_fn,
        is_distributional=True,
        n_outer_folds=CONFIG["n_outer_folds"],
        n_inner_folds=CONFIG["n_inner_folds"],
        epochs=CONFIG["epochs"],
        patience=CONFIG["patience"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
        device=device,
        min_tiles_within_patient=CONFIG["min_tiles_within_patient"],
        n_bootstrap=CONFIG["n_bootstrap"],
        n_permutations=CONFIG["n_permutations"],
        grad_clip_norm=CONFIG["grad_clip_norm"],
        return_attention=True,
    )

    predictions_df = cv_results["predictions"]
    metrics_df = cv_results["metrics_summary"]
    per_patient_df = cv_results["per_patient_metrics"]
    fold_df = cv_results["fold_assignments"]
    calibration_df = cv_results["calibration_summary"]
    selected_epochs = cv_results["selected_epochs"]
    all_attentions = cv_results["attention_weights"]

    print(f"\nOuter fold selected epochs: {selected_epochs}")

    # === persist primary CV outputs
    predictions_df.to_csv(output_dir / "cv_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    per_patient_df.to_csv(output_dir / "cv_per_patient_metrics.csv", index=False)
    fold_df.to_csv(output_dir / "cv_fold_assignments.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_summary.csv", index=False)

    # === compute and persist attention heterogeneity diagnostics
    print("Computing tile-level attention diagnostics (entropy, Gini, top-k mass, skewness)...")
    diag_records = []
    attention_dict = {}

    for i in range(len(groups)):
        pid = groups[i]
        tid = tile_ids[i]
        w = all_attentions[i]
        diag = compute_tile_diagnostics(w, top_k=CONFIG["top_k_attention"])
        diag_records.append({
            "patient_id": pid,
            "tile_id": tid,
            **diag,
        })
        attention_dict[f"{pid}_{tid}"] = w.astype(np.float32)

    diag_df = pd.DataFrame(diag_records)
    diag_df.to_csv(output_dir / "attention_diagnostics.csv", index=False)

    # Persist compressed attention weights
    np.savez_compressed(output_dir / "attention_weights.npz", **attention_dict)

    # === display outer-fold test results
    print(f"\n{'='*118}")
    print(f"=== PHASE 2 ABMIL OUTER-FOLD TEST RESULTS ({n_tiles} tiles, {n_patients} patients) | 95% CI: bootstrap ===")
    print(f"{'='*118}")
    for _, m in metrics_df.iterrows():
        print(f"{m['target']:<11}| pooled r {m['pooled_r']:+.3f} [{m['pooled_r_ci_low']:+.3f}, {m['pooled_r_ci_high']:+.3f}] "
              f"| within r {m['within_patient_r']:+.3f} [{m['within_patient_r_ci_low']:+.3f}, {m['within_patient_r_ci_high']:+.3f}] p={m['within_patient_p']:.1e} "
              f"| between r {m['between_patient_r']:+.3f} [{m['between_patient_r_ci_low']:+.3f}, {m['between_patient_r_ci_high']:+.3f}] p={m['between_patient_p']:.1e} "
              f"| AUC {m['high_vs_low_auc']:.3f}")
    print(f"{'='*118}")

    print(f"\n{'='*95}")
    print("=== UNCERTAINTY (SIGMA) CALIBRATION SUMMARY ===")
    print(f"{'='*95}")
    for _, c in calibration_df.iterrows():
        print(f"{c['target']:<11}| 1-sigma coverage: {c['coverage_1sigma_pct']:.1f}% (target ~68%) "
              f"| 2-sigma: {c['coverage_2sigma_pct']:.1f}% (target ~95%) "
              f"| error vs std Spearman rho: {c['error_vs_std_spearman']:+.3f} (p={c['error_vs_std_p']:.1e})")
    print(f"{'='*95}")

    print(f"\n{'='*95}")
    print("=== ATTENTION HETEROGENEITY READOUT SUMMARY ===")
    print(f"{'='*95}")
    print(f"Mean Normalized Attention Entropy (H_norm) : {diag_df['normalized_attention_entropy'].mean():.3f} +/- {diag_df['normalized_attention_entropy'].std():.3f}")
    print(f"Mean Gini Coefficient (G)                  : {diag_df['gini_coefficient'].mean():.3f} +/- {diag_df['gini_coefficient'].std():.3f}")
    print(f"Mean Top-{CONFIG['top_k_attention']} Attention Mass (R_top{CONFIG['top_k_attention']})      : {diag_df['topk_attention_mass'].mean():.3f} +/- {diag_df['topk_attention_mass'].std():.3f}")
    print(f"Mean Attention Skewness                    : {diag_df['attention_skewness'].mean():.3f} +/- {diag_df['attention_skewness'].std():.3f}")
    print(f"{'='*95}\n")

    # === run summary
    finished_at = datetime.now().astimezone()
    summary_info = {
        "analysis_finished_at": finished_at.isoformat(timespec="seconds"),
        "analysis_started_at": started_at.isoformat(timespec="seconds"),
        "git": git_state(),
        "phase": "phase2_abmil",
        "model": "abmil_distributional",
        "loss": f"gaussian_beta_nll (beta={CONFIG['beta']})",
        "features": CONFIG["model_name"],
        "targets": SIGNATURES,
        "n_patients": int(n_patients),
        "n_tiles": int(n_tiles),
        "selected_epochs_per_outer_fold": selected_epochs,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in CONFIG.items()},
        "scaling": "features standardized across training patches and targets z-scored with training-fold statistics only",
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

    print(f"Finished at {summary_info['analysis_finished_at']}. All outputs written to: {output_dir}")


if __name__ == "__main__":
    run()

