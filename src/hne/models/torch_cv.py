"""
/src/hne/models/torch_cv.py

Modular, leak-free PyTorch nested grouped (by patient) cross-validation harness.
Designed for both mean-pooled vector inputs (Phase 1b) and unpooled patch-level embeddings (Phase 2 ABMIL).

Ensures:
  1. Outer GroupKFold by patient (every patient tested exactly once out-of-fold).
  2. Inner GroupKFold by patient on outer-training patients for early stopping / epoch selection.
  3. Leak-free cohort z-scoring on training split only.
  4. Out-of-fold predictions back-transformed to raw score units.
  5. Full Phase 1 & Rule 8 compliance: patient-level bootstrap CIs, permutations, and calibration.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hne.models.evaluation import evaluate_target


class TileDataset(Dataset):
    """
    Unified dataset supporting:
      - 2D feature matrix (N_tiles, d): for mean-pooled embeddings (Phase 1b)
      - List of patch matrices [ (M_i, d) ]: for unpooled patch embeddings (Phase 2 ABMIL)
    """

    def __init__(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        y: np.ndarray,
        x_scaler: Optional[StandardScaler] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ):
        self.is_bag = isinstance(X, list)
        
        # Scale targets if statistics are provided
        if y_mean is not None and y_std is not None:
            scaled_y = (y - y_mean) / y_std
        else:
            scaled_y = y
        self.y = torch.from_numpy(np.asarray(scaled_y)).float()

        # Scale features if scaler is provided
        if self.is_bag:
            if x_scaler is not None:
                self.X = [torch.from_numpy(x_scaler.transform(x)).float() for x in X]
            else:
                self.X = [torch.from_numpy(x).float() for x in X]
        else:
            if x_scaler is not None:
                scaled_x = x_scaler.transform(X)
            else:
                scaled_x = X
            self.X = torch.from_numpy(scaled_x).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def collate_tile_bags(batch):
    """
    Handles variable-sized patch bags.
    If inputs are 1D vectors, uses default stacking.
    If inputs are 2D patch matrices, returns list of tensors and stacked targets.
    """
    samples, targets = zip(*batch)
    if samples[0].ndim == 1:
        # Standard vector embeddings (Phase 1b)
        return torch.stack(samples, dim=0), torch.stack(targets, dim=0)
    else:
        # Patch matrices of shape (M_i, d) (Phase 2 ABMIL)
        return list(samples), torch.stack(targets, dim=0)


def fit_scalers(
    X_tr: Union[np.ndarray, List[np.ndarray]],
    y_tr: np.ndarray,
) -> Tuple[Optional[StandardScaler], np.ndarray, np.ndarray]:
    """
    Computes feature and target statistics strictly from the training split.
    If X_tr is a 2D feature matrix, fits a StandardScaler.
    If X_tr is a list of variable-size patch matrices, feature scaling is computed
    across all training patches.
    """
    if isinstance(X_tr, np.ndarray) and X_tr.ndim == 2:
        x_scaler = StandardScaler().fit(X_tr)
    elif isinstance(X_tr, list):
        all_patches = np.concatenate(X_tr, axis=0)
        x_scaler = StandardScaler().fit(all_patches)
    else:
        x_scaler = None

    y_mean = y_tr.mean(axis=0)
    y_std = y_tr.std(axis=0)
    y_std[y_std == 0] = 1.0
    return x_scaler, y_mean, y_std


def compute_calibration_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates Rule 8 uncertainty calibration per target:
      1. Coverage 1-sigma: % of true values in [mu - sigma, mu + sigma] (nominal 68.3%)
      2. Coverage 2-sigma: % of true values in [mu - 2*sigma, mu + 2*sigma] (nominal 95.4%)
      3. Error-vs-sigma correlation: Spearman rank correlation between |y_true - mu_pred| and std_pred
    """
    records = []
    for sig in predictions_df["target"].unique():
        sub = predictions_df[predictions_df["target"] == sig].copy()
        y_true = sub["y_true"].values
        mu_pred = sub["mu_pred"].values
        std_pred = sub["std_pred"].values

        abs_err = np.abs(y_true - mu_pred)
        cov_1sig = float(np.mean(abs_err <= std_pred) * 100.0)
        cov_2sig = float(np.mean(abs_err <= (2.0 * std_pred)) * 100.0)

        if np.std(abs_err) > 1e-6 and np.std(std_pred) > 1e-6:
            rho, p_val = spearmanr(abs_err, std_pred)
        else:
            rho, p_val = 0.0, 1.0

        records.append({
            "target": sig,
            "coverage_1sigma_pct": cov_1sig,
            "coverage_2sigma_pct": cov_2sig,
            "error_vs_std_spearman": float(rho),
            "error_vs_std_p": float(p_val),
            "mean_predicted_std": float(np.mean(std_pred)),
            "mean_abs_error": float(np.mean(abs_err)),
        })
    return pd.DataFrame(records)


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    patience: int,
    device: torch.device,
    is_distributional: bool = True,
    grad_clip_norm: float = 1.0,
) -> int:
    """
    Trains a model with early stopping on validation loss. Returns the best epoch.
    """
    best_loss = float("inf")
    best_weights = None
    best_epoch = 1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            if isinstance(bx, torch.Tensor):
                bx = bx.to(device)
            elif isinstance(bx, list):
                bx = [x.to(device) for x in bx]
            by = by.to(device)

            optimizer.zero_grad()
            if is_distributional:
                mu, std = model(bx)
                loss = loss_fn(mu, std, by)
            else:
                mu = model(bx)
                loss = loss_fn(mu, by)

            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        # Validation check
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                if isinstance(vx, torch.Tensor):
                    vx = vx.to(device)
                elif isinstance(vx, list):
                    vx = [x.to(device) for x in vx]
                vy = vy.to(device)

                if is_distributional:
                    v_mu, v_std = model(vx)
                    v_loss = loss_fn(v_mu, v_std, vy)
                else:
                    v_mu = model(vx)
                    v_loss = loss_fn(v_mu, vy)
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


def nested_cv_torch(
    X: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray,
    tile_ids: np.ndarray,
    target_names: List[str],
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    loss_fn: nn.Module,
    is_distributional: bool = True,
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    epochs: int = 70,
    patience: int = 15,
    batch_size: int = 64,
    seed: int = 42,
    device: Optional[torch.device] = None,
    min_tiles_within_patient: int = 5,
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    grad_clip_norm: float = 1.0,
    return_attention: bool = False,
) -> Dict[str, Union[pd.DataFrame, Dict, List]]:
    """
    Executes leak-free nested grouped cross-validation for a PyTorch model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_bag = isinstance(X, list)
    n_samples = len(X)
    outer = GroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    y_pred_mu = np.full_like(y, np.nan, dtype=np.float64)
    y_pred_std = np.full_like(y, np.nan, dtype=np.float64) if is_distributional else None
    fold_of_tile = np.full(n_samples, -1)
    selected_epochs = {}
    all_attentions = [None] * n_samples if return_attention else None

    indices = np.arange(n_samples)

    for outer_fold, (tr, te) in enumerate(outer.split(indices, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), f"Patient leak in outer fold {outer_fold}"

        # Inner grouped CV on outer training patients
        inner = GroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed + 1 + outer_fold)
        inner_tr_sub, inner_va_sub = next(inner.split(tr, y[tr], groups[tr]))
        inner_tr = tr[inner_tr_sub]
        inner_va = tr[inner_va_sub]

        X_tr = [X[i] for i in tr] if is_bag else X[tr]
        X_in_tr = [X[i] for i in inner_tr] if is_bag else X[inner_tr]
        X_in_va = [X[i] for i in inner_va] if is_bag else X[inner_va]
        X_te = [X[i] for i in te] if is_bag else X[te]

        # Inner scalers
        x_scaler_in, y_mean_in, y_std_in = fit_scalers(X_in_tr, y[inner_tr])
        ds_in_tr = TileDataset(X_in_tr, y[inner_tr], x_scaler=x_scaler_in, y_mean=y_mean_in, y_std=y_std_in)
        ds_in_va = TileDataset(X_in_va, y[inner_va], x_scaler=x_scaler_in, y_mean=y_mean_in, y_std=y_std_in)

        in_train_loader = DataLoader(ds_in_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_tile_bags)
        in_val_loader = DataLoader(ds_in_va, batch_size=batch_size, shuffle=False, collate_fn=collate_tile_bags)

        inner_model = model_factory().to(device)
        opt_in = optimizer_factory(inner_model)

        best_epoch = train_single_model(
            model=inner_model,
            train_loader=in_train_loader,
            val_loader=in_val_loader,
            optimizer=opt_in,
            loss_fn=loss_fn,
            epochs=epochs,
            patience=patience,
            device=device,
            is_distributional=is_distributional,
            grad_clip_norm=grad_clip_norm,
        )
        selected_epochs[outer_fold] = best_epoch

        # Refit model on ALL outer training tiles up to selected best_epoch
        x_scaler_out, y_mean_out, y_std_out = fit_scalers(X_tr, y[tr])
        ds_out_tr = TileDataset(X_tr, y[tr], x_scaler=x_scaler_out, y_mean=y_mean_out, y_std=y_std_out)
        out_train_loader = DataLoader(ds_out_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_tile_bags)

        final_model = model_factory().to(device)
        opt_out = optimizer_factory(final_model)

        final_model.train()
        for ep in range(1, best_epoch + 1):
            for bx, by in out_train_loader:
                if isinstance(bx, torch.Tensor):
                    bx = bx.to(device)
                elif isinstance(bx, list):
                    bx = [x.to(device) for x in bx]
                by = by.to(device)

                opt_out.zero_grad()
                if is_distributional:
                    mu, std = final_model(bx)
                    loss = loss_fn(mu, std, by)
                else:
                    mu = final_model(bx)
                    loss = loss_fn(mu, by)

                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=grad_clip_norm)
                opt_out.step()

        # Predict outer test tiles
        final_model.eval()
        ds_out_te = TileDataset(X_te, y[te], x_scaler=x_scaler_out, y_mean=y_mean_out, y_std=y_std_out)
        te_loader = DataLoader(ds_out_te, batch_size=batch_size, shuffle=False, collate_fn=collate_tile_bags)

        fold_preds_mu = []
        fold_preds_std = []
        fold_attns = []
        with torch.no_grad():
            for vx, _ in te_loader:
                if isinstance(vx, torch.Tensor):
                    vx = vx.to(device)
                elif isinstance(vx, list):
                    vx = [x.to(device) for x in vx]

                if return_attention:
                    if is_distributional:
                        mu_norm, std_norm, attns = final_model(vx, return_attention=True)
                        fold_preds_mu.append(mu_norm.cpu().numpy())
                        fold_preds_std.append(std_norm.cpu().numpy())
                        fold_attns.extend(attns)
                    else:
                        mu_norm, attns = final_model(vx, return_attention=True)
                        fold_preds_mu.append(mu_norm.cpu().numpy())
                        fold_attns.extend(attns)
                else:
                    if is_distributional:
                        mu_norm, std_norm = final_model(vx)
                        fold_preds_mu.append(mu_norm.cpu().numpy())
                        fold_preds_std.append(std_norm.cpu().numpy())
                    else:
                        mu_norm = final_model(vx)
                        fold_preds_mu.append(mu_norm.cpu().numpy())

        mu_norm_all = np.concatenate(fold_preds_mu, axis=0)
        y_pred_mu[te] = mu_norm_all * y_std_out + y_mean_out
        if is_distributional:
            std_norm_all = np.concatenate(fold_preds_std, axis=0)
            y_pred_std[te] = std_norm_all * y_std_out

        if return_attention:
            for idx_in_te, sample_idx in enumerate(te):
                all_attentions[sample_idx] = fold_attns[idx_in_te]

        fold_of_tile[te] = outer_fold

    assert not np.isnan(y_pred_mu).any() and (fold_of_tile >= 0).all(), "Unpredicted tile detected"

    # Assemble prediction DataFrame
    pred_records = []
    for k, sig in enumerate(target_names):
        df_sig = pd.DataFrame({
            "patient_id": groups,
            "tile_id": tile_ids,
            "outer_fold": fold_of_tile,
            "target": sig,
            "y_true": y[:, k],
            "mu_pred": y_pred_mu[:, k],
        })
        if is_distributional:
            df_sig["std_pred"] = y_pred_std[:, k]
        pred_records.append(df_sig)
    predictions_df = pd.concat(pred_records, ignore_index=True)

    # Fold assignments
    fold_df = (
        pd.DataFrame({"patient_id": groups, "outer_fold": fold_of_tile})
        .groupby("patient_id")
        .agg(outer_fold=("outer_fold", "first"), n_tiles=("outer_fold", "size"))
        .reset_index()
    )

    # Patient-level evaluation across all targets
    metric_rows, per_patient_frames = [], []
    for k, sig in enumerate(target_names):
        metrics, per_patient = evaluate_target(
            y_true=y[:, k].astype(np.float64),
            y_pred=y_pred_mu[:, k],
            patient_ids=groups,
            min_tiles_within=min_tiles_within_patient,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed,
        )
        metric_rows.append({"target": sig, **metrics})
        per_patient.insert(0, "target", sig)
        per_patient_frames.append(per_patient)

    metrics_df = pd.DataFrame(metric_rows)
    per_patient_df = pd.concat(per_patient_frames, ignore_index=True)

    calibration_df = compute_calibration_metrics(predictions_df) if is_distributional else None

    return {
        "metrics_summary": metrics_df,
        "predictions": predictions_df,
        "per_patient_metrics": per_patient_df,
        "fold_assignments": fold_df,
        "calibration_summary": calibration_df,
        "selected_epochs": selected_epochs,
        "attention_weights": all_attentions,
    }

