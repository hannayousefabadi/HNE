"""scripts/model_train/phase1b/train_phase1b_distributional.py

Phase 1b: Distributional MLP regression on mean-pooled Phikon-v2 tile embeddings.
Predicts both conditional mean (mu) and uncertainty (sigma) per pathway signature score.
Evaluated with nested grouped (by patient) cross-validation and patient-level statistics.
"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from hne.core.config import ROOT
from hne.core.paths import PATIENT_IDS, RESULTS
from hne.models.data import FEATURE_REGISTRY, get_available_patient_ids, load_features_and_targets
from hne.models.evaluation import AUC_DEFINITION, evaluate_target
from hne.models.mlp import DistributionalMLP, GaussianBetaNLLLoss

CONFIG = {
    "model_name": "phikon_v2",
    "output_dir": RESULTS / "phase1b_distributional",
    "n_outer_folds": 5,
    "n_inner_folds": 5,
    "hidden_dim": 128,
    "dropout_rate": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 70,
    "batch_size": 64,
    "patience": 15,
    "beta": 0.5,                        # beta-weighting for stabilized Gaussian NLL
    "seed": 42,
    "min_tiles_within_patient": 5,      # patients with fewer tiles get no within-patient r
    "n_bootstrap": 2000,
    "n_permutations": 10000,
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


def fit_scalers(X_tr: np.ndarray, y_tr: np.ndarray):
    """Leak-free feature and target statistics from training split only."""
    x_scaler = StandardScaler().fit(X_tr)
    y_mean = y_tr.mean(axis=0)
    y_std = y_tr.std(axis=0)
    y_std[y_std == 0] = 1.0
    return x_scaler, y_mean, y_std


def train_single_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    patience: int,
    device: torch.device,
) -> int:
    """Trains with early stopping on inner validation loss. Returns best epoch."""
    best_loss = float("inf")
    best_weights = None
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            mu, std = model(bx)
            loss = loss_fn(mu, std, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation check
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_mu, v_std = model(vx)
                v_loss = loss_fn(v_mu, v_std, vy)
                val_losses.append(v_loss.item())

        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_epoch


def compute_calibration_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates Rule 8 uncertainty calibration per signature:
      1. Empirical coverage: % of true values in [mu - sigma, mu + sigma] (nominal 68.3%)
      2. 2-sigma coverage: % of true values in [mu - 2*sigma, mu + 2*sigma] (nominal 95.4%)
      3. Error-uncertainty correlation: Spearman correlation between |y_true - mu_pred| and std_pred
    """
    records = []
    for sig in predictions_df["target"].unique():
        sub = predictions_df[predictions_df["target"] == sig].copy()
        y_true = sub["y_true"].values
        mu_pred = sub["mu_pred"].values
        std_pred = sub["std_pred"].values

        abs_err = np.abs(y_true - mu_pred)
        cov_1sig = np.mean(abs_err <= std_pred) * 100.0
        cov_2sig = np.mean(abs_err <= (2.0 * std_pred)) * 100.0

        if np.std(abs_err) > 1e-6 and np.std(std_pred) > 1e-6:
            rho, p_val = spearmanr(abs_err, std_pred)
        else:
            rho, p_val = 0.0, 1.0

        records.append({
            "target": sig,
            "coverage_1sigma_pct": cov_1sig,
            "coverage_2sigma_pct": cov_2sig,
            "error_vs_std_spearman": rho,
            "error_vs_std_p": p_val,
            "mean_predicted_std": float(np.mean(std_pred)),
            "mean_abs_error": float(np.mean(abs_err)),
        })
    return pd.DataFrame(records)


def run():
    started_at = datetime.now().astimezone()
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_spec = FEATURE_REGISTRY[CONFIG["model_name"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPhase 1b Distributional MLP Training | Device: {device}")

    # === cohort discovery and loading
    patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    X, y, tile_meta = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=patients,
        target_cols=TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
    )
    groups = np.array([pid for pid, _ in tile_meta])
    tile_ids = np.array([tid for _, tid in tile_meta])
    n_targets = len(TARGET_COLS)
    input_dim = X.shape[1]

    print(f"Cohort: {len(np.unique(groups))} patients, {len(X)} tiles, {input_dim}-d features")
    print(f"Nested grouped CV: {CONFIG['n_outer_folds']} outer x {CONFIG['n_inner_folds']} inner folds, seed {CONFIG['seed']}")

    # === outer CV: every patient lands in exactly one outer test fold
    outer = GroupKFold(n_splits=CONFIG["n_outer_folds"], shuffle=True, random_state=CONFIG["seed"])
    y_pred_mu = np.full_like(y, np.nan, dtype=np.float64)
    y_pred_std = np.full_like(y, np.nan, dtype=np.float64)
    fold_of_tile = np.full(len(y), -1)

    for outer_fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), "patient leaked across outer folds"
        print(f"\n--- Outer Fold {outer_fold}: {len(np.unique(groups[tr]))} train / {len(np.unique(groups[te]))} test patients ---")

        # Inner grouped CV to select best epoch / early stopping without touching outer test
        inner = GroupKFold(n_splits=CONFIG["n_inner_folds"], shuffle=True, random_state=CONFIG["seed"] + 1 + outer_fold)
        inner_tr, inner_va = next(inner.split(X[tr], y[tr], groups[tr]))

        # Inner scalers
        x_scaler_in, y_mean_in, y_std_in = fit_scalers(X[tr][inner_tr], y[tr][inner_tr])
        X_in_tr = torch.from_numpy(x_scaler_in.transform(X[tr][inner_tr])).float()
        y_in_tr = torch.from_numpy((y[tr][inner_tr] - y_mean_in) / y_std_in).float()
        X_in_va = torch.from_numpy(x_scaler_in.transform(X[tr][inner_va])).float()
        y_in_va = torch.from_numpy((y[tr][inner_va] - y_mean_in) / y_std_in).float()

        in_train_loader = DataLoader(TensorDataset(X_in_tr, y_in_tr), batch_size=CONFIG["batch_size"], shuffle=True)
        in_val_loader = DataLoader(TensorDataset(X_in_va, y_in_va), batch_size=CONFIG["batch_size"], shuffle=False)

        inner_model = DistributionalMLP(
            input_dim=input_dim,
            n_targets=n_targets,
            hidden_dim=CONFIG["hidden_dim"],
            dropout_rate=CONFIG["dropout_rate"],
        ).to(device)

        opt_in = torch.optim.AdamW(inner_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        loss_fn = GaussianBetaNLLLoss(beta=CONFIG["beta"])

        best_epoch = train_single_model(
            model=inner_model,
            train_loader=in_train_loader,
            val_loader=in_val_loader,
            optimizer=opt_in,
            loss_fn=loss_fn,
            epochs=CONFIG["epochs"],
            patience=CONFIG["patience"],
            device=device,
        )
        print(f"Outer fold {outer_fold}: inner CV selected stopping epoch = {best_epoch}")

        # Refit model on ALL outer training tiles up to selected best_epoch
        x_scaler_out, y_mean_out, y_std_out = fit_scalers(X[tr], y[tr])
        X_out_tr = torch.from_numpy(x_scaler_out.transform(X[tr])).float()
        y_out_tr = torch.from_numpy((y[tr] - y_mean_out) / y_std_out).float()
        out_train_loader = DataLoader(TensorDataset(X_out_tr, y_out_tr), batch_size=CONFIG["batch_size"], shuffle=True)

        final_model = DistributionalMLP(
            input_dim=input_dim,
            n_targets=n_targets,
            hidden_dim=CONFIG["hidden_dim"],
            dropout_rate=CONFIG["dropout_rate"],
        ).to(device)
        opt_out = torch.optim.AdamW(final_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

        final_model.train()
        for ep in range(1, best_epoch + 1):
            for bx, by in out_train_loader:
                bx, by = bx.to(device), by.to(device)
                opt_out.zero_grad()
                mu, std = final_model(bx)
                loss = loss_fn(mu, std, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
                opt_out.step()

        # Predict outer test tiles
        final_model.eval()
        X_out_te = torch.from_numpy(x_scaler_out.transform(X[te])).float().to(device)
        with torch.no_grad():
            mu_norm, std_norm = final_model(X_out_te)
            mu_norm = mu_norm.cpu().numpy()
            std_norm = std_norm.cpu().numpy()

        # Back-transform predictions to raw units
        # mu_raw = mu_norm * y_std + y_mean
        # std_raw = std_norm * y_std
        y_pred_mu[te] = mu_norm * y_std_out + y_mean_out
        y_pred_std[te] = std_norm * y_std_out
        fold_of_tile[te] = outer_fold

    assert not np.isnan(y_pred_mu).any() and (fold_of_tile >= 0).all(), "a tile was never predicted"

    # === persist predictions
    predictions_df = pd.concat([
        pd.DataFrame({
            "patient_id": groups,
            "tile_id": tile_ids,
            "outer_fold": fold_of_tile,
            "target": sig,
            "y_true": y[:, k],
            "mu_pred": y_pred_mu[:, k],
            "std_pred": y_pred_std[:, k],
        }) for k, sig in enumerate(SIGNATURES)
    ], ignore_index=True)
    predictions_df.to_csv(output_dir / "cv_predictions.csv", index=False)

    (pd.DataFrame({"patient_id": groups, "outer_fold": fold_of_tile})
       .groupby("patient_id").agg(outer_fold=("outer_fold", "first"), n_tiles=("outer_fold", "size"))
       .reset_index().to_csv(output_dir / "cv_fold_assignments.csv", index=False))

    # === patient-level evaluation, all signatures
    metric_rows, per_patient_frames = [], []
    for k, sig in enumerate(SIGNATURES):
        metrics, per_patient = evaluate_target(
            y_true=y[:, k].astype(np.float64),
            y_pred=y_pred_mu[:, k],
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

    # === evaluate uncertainty calibration
    calib_df = compute_calibration_metrics(predictions_df)
    calib_df.to_csv(output_dir / "calibration_summary.csv", index=False)

    print(f"\n{'='*118}")
    print(f"=== OUTER-FOLD TEST RESULTS ({len(X)} tiles, {len(np.unique(groups))} patients) | 95% CI: patient-level bootstrap ===")
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
    for _, c in calib_df.iterrows():
        print(f"{c['target']:<11}| 1-sigma coverage: {c['coverage_1sigma_pct']:.1f}% (target ~68%) "
              f"| 2-sigma: {c['coverage_2sigma_pct']:.1f}% (target ~95%) "
              f"| error vs std Spearman rho: {c['error_vs_std_spearman']:+.3f} (p={c['error_vs_std_p']:.1e})")
    print(f"{'='*95}\n")

    # === run summary
    finished_at = datetime.now().astimezone()
    summary_info = {
        "analysis_finished_at": finished_at.isoformat(timespec="seconds"),
        "analysis_started_at": started_at.isoformat(timespec="seconds"),
        "git": git_state(),
        "model": "distributional_mlp",
        "loss": f"gaussian_beta_nll (beta={CONFIG['beta']})",
        "features": CONFIG["model_name"],
        "targets": SIGNATURES,
        "n_patients": int(len(np.unique(groups))),
        "n_tiles": int(len(X)),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in CONFIG.items()},
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

