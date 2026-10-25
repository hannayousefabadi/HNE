"""
tests/test_phase2_smoke.py

Fast end-to-end smoke test for Phase 2 ABMIL on a 3-patient subset with unpooled patch features.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hne.core.paths import PATIENT_IDS
from hne.models.abmil import ABMIL, compute_tile_diagnostics
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.mlp import GaussianBetaNLLLoss
from hne.models.torch_cv import nested_cv_torch

SIGNATURES = ["FMRP", "Cell_cycle", "YAP", "WNT", "EMT"]
TARGET_COLS = [f"{sig}_signature_score" for sig in SIGNATURES]


def run_smoke_test():
    print("=" * 70)
    print("RUNNING PHASE 2 ABMIL FAST SMOKE TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_spec = FEATURE_REGISTRY["phikon_v2_unpooled"]
    all_patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    test_patients = all_patients[:6]  # take first 6 patients to have >= 3 train patients per outer fold

    print(f"Loading unpooled patch bags for test patients: {test_patients}")
    X, y, tile_meta = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=test_patients,
        target_cols=TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
        as_bags=True,
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])

    print(f"Loaded {len(X)} tiles from {len(test_patients)} patients.")

    def model_factory():
        return ABMIL(
            input_dim=1024,
            n_targets=len(SIGNATURES),
            attention_dim=64,
            head_hidden_dim=64,
            dropout_rate=0.2,
            is_distributional=True,
        )

    def optimizer_factory(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    loss_fn = GaussianBetaNLLLoss(beta=0.5)

    print("Running nested_cv_torch (2 outer x 2 inner folds, 2 epochs)...")
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
        n_outer_folds=2,
        n_inner_folds=2,
        epochs=2,
        patience=2,
        batch_size=8,
        seed=42,
        device=device,
        min_tiles_within_patient=3,
        n_bootstrap=10,
        n_permutations=10,
        grad_clip_norm=1.0,
        return_attention=True,
    )

    preds = results["predictions"]
    metrics = results["metrics_summary"]
    calib = results["calibration_summary"]
    attns = results["attention_weights"]

    assert len(preds) == len(X) * len(SIGNATURES), f"Prediction count mismatch: {len(preds)}"
    assert len(attns) == len(X), f"Attention bag count mismatch: {len(attns)}"
    assert metrics is not None and len(metrics) == len(SIGNATURES)
    assert calib is not None and len(calib) == len(SIGNATURES)

    # Compute diagnostics on all tiles
    for i, w in enumerate(attns):
        diag = compute_tile_diagnostics(w, top_k=5)
        assert 0.0 <= diag["normalized_attention_entropy"] <= 1.0
        assert 0.0 <= diag["gini_coefficient"] <= 1.0

    print("\n[SUCCESS] Phase 2 ABMIL fast smoke test passed completely!")


if __name__ == "__main__":
    run_smoke_test()
