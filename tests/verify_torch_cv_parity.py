"""
tests/verify_torch_cv_parity.py

Verifies parity between src/hne/models/torch_cv.py and the Phase 1b Distributional MLP baseline.
Runs Phase 1b configuration through nested_cv_torch and compares metrics against
results/phase1b_distributional/cv_metrics_summary.csv and calibration_summary.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hne.core.paths import PATIENT_IDS, RESULTS
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.mlp import DistributionalMLP, GaussianBetaNLLLoss
from hne.models.torch_cv import nested_cv_torch

SIGNATURES = ["FMRP", "Cell_cycle", "YAP", "WNT", "EMT"]
TARGET_COLS = [f"{sig}_signature_score" for sig in SIGNATURES]


def run_verification():
    print("=" * 80)
    print("VERIFYING torch_cv.py PARITY ON PHASE 1b DISTRIBUTIONAL MLP")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    feature_spec = FEATURE_REGISTRY["phikon_v2"]
    patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    X, y, tile_meta = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=patients,
        target_cols=TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])
    input_dim = X.shape[1]
    n_targets = len(TARGET_COLS)

    print(f"Data: {len(np.unique(groups))} patients, {len(X)} tiles, {input_dim}-d features")

    # Model and optimizer factories
    def model_factory():
        return DistributionalMLP(
            input_dim=input_dim,
            n_targets=n_targets,
            hidden_dim=128,
            dropout_rate=0.2,
        )

    def optimizer_factory(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    loss_fn = GaussianBetaNLLLoss(beta=0.5)

    print("\nRunning nested_cv_torch...")
    results = nested_cv_torch(
        X=X,
        y=y,
        groups=groups,
        tile_ids=tile_ids,
        target_names=SIGNATURES,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        loss_fn=loss_fn,
        is_distributional=True,
        n_outer_folds=5,
        n_inner_folds=5,
        epochs=70,
        patience=15,
        batch_size=64,
        seed=42,
        device=device,
        min_tiles_within_patient=5,
        n_bootstrap=2000,
        n_permutations=10000,
        grad_clip_norm=1.0,
    )

    new_metrics = results["metrics_summary"]
    new_calib = results["calibration_summary"]
    selected_epochs = results["selected_epochs"]

    print("\nSelected epochs per outer fold:", selected_epochs)

    # Load Phase 1b baseline
    phase1b_dir = RESULTS / "phase1b_distributional"
    ref_metrics = pd.read_csv(phase1b_dir / "cv_metrics_summary.csv")
    ref_calib = pd.read_csv(phase1b_dir / "calibration_summary.csv")

    print("\n" + "=" * 90)
    print("METRICS COMPARISON: torch_cv.py vs Phase 1b Frozen Baseline")
    print("=" * 90)
    print(f"{'Target':<12} | {'Metric':<18} | {'Phase 1b (Frozen)':<18} | {'torch_cv Run':<18} | {'Diff':<10}")
    print("-" * 90)

    max_within_diff = 0.0
    for sig in SIGNATURES:
        ref_row = ref_metrics[ref_metrics["target"] == sig].iloc[0]
        new_row = new_metrics[new_metrics["target"] == sig].iloc[0]

        for metric in ["within_patient_r", "pooled_r", "between_patient_r", "high_vs_low_auc"]:
            ref_val = ref_row[metric]
            new_val = new_row[metric]
            diff = new_val - ref_val
            if metric == "within_patient_r":
                max_within_diff = max(max_within_diff, abs(diff))
            print(f"{sig:<12} | {metric:<18} | {ref_val:+.4f}             | {new_val:+.4f}             | {diff:+.4f}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("CALIBRATION COMPARISON: torch_cv.py vs Phase 1b Frozen Baseline")
    print("=" * 90)
    print(f"{'Target':<12} | {'Phase 1b Cov1s (%)':<20} | {'torch_cv Cov1s (%)':<20} | {'Phase 1b rho':<15} | {'torch_cv rho':<15}")
    print("-" * 90)
    for sig in SIGNATURES:
        ref_c = ref_calib[ref_calib["target"] == sig].iloc[0]
        new_c = new_calib[new_calib["target"] == sig].iloc[0]
        print(f"{sig:<12} | {ref_c['coverage_1sigma_pct']:<20.1f} | {new_c['coverage_1sigma_pct']:<20.1f} | {ref_c['error_vs_std_spearman']:<15.3f} | {new_c['error_vs_std_spearman']:<15.3f}")
    print("=" * 90)

    print(f"\nMax |difference| in within_patient_r: {max_within_diff:.4f}")
    # Threshold for parity given stochastic neural training
    assert max_within_diff < 0.05, f"Discrepancy too large ({max_within_diff:.4f} >= 0.05)"
    print("\n[SUCCESS] torch_cv.py verified on Phase 1b! Parity confirmed.\n")


if __name__ == "__main__":
    run_verification()

