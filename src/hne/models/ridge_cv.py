"""
/src/hne/models/ridge_cv.py

Nested grouped (by patient) cross-validation for ridge regression. Shared by the Phase 1
baseline and by control analyses, so they all run through the same harness.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def fit_scalers(X: np.ndarray, y: np.ndarray):
    """Feature scaler and target mean/std from one training fold only."""
    x_scaler = StandardScaler().fit(X)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std == 0] = 1.0
    return x_scaler, y_mean, y_std


def select_alphas(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: np.ndarray,
                  n_inner_folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Inner grouped CV on the outer-training patients. Returns, per target, the alpha with
    the lowest mean squared error over all inner-validation tiles, and the full error grid.
    """
    sq_err = np.zeros((len(alphas), y.shape[1]))
    inner = GroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)

    for tr, va in inner.split(X, y, groups):
        x_scaler, y_mean, y_std = fit_scalers(X[tr], y[tr])
        X_tr, X_va = x_scaler.transform(X[tr]), x_scaler.transform(X[va])
        z_tr, z_va = (y[tr] - y_mean) / y_std, (y[va] - y_mean) / y_std
        for a_idx, alpha in enumerate(alphas):
            pred = Ridge(alpha=alpha).fit(X_tr, z_tr).predict(X_va).reshape(z_va.shape)   # 1-d when there is one target
            sq_err[a_idx] += ((pred - z_va) ** 2).sum(axis=0)

    mse = sq_err / len(y)
    return alphas[mse.argmin(axis=0)], mse


def nested_cv_ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: np.ndarray,
                    n_outer_folds: int, n_inner_folds: int, seed: int):
    """
    Every patient lands in exactly one outer test fold. For each outer fold, alpha is chosen
    per target by inner CV on the training patients, the model is refit on all of them, and
    the test tiles are predicted in raw target units.

    Returns:
        y_pred: (n_tiles, n_targets) out-of-fold predictions
        fold_of_tile: (n_tiles,) outer fold index of each tile
        fold_records: one dict per outer fold and target (selected alpha, inner MSE, fold sizes)
    """
    outer = GroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    y_pred = np.full_like(y, np.nan, dtype=np.float64)
    fold_of_tile = np.full(len(y), -1)
    fold_records = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        assert not set(groups[tr]) & set(groups[te]), "patient leaked across outer folds"

        best_alphas, inner_mse = select_alphas(X[tr], y[tr], groups[tr], alphas, n_inner_folds, seed=seed + 1 + fold)

        # refit on all outer-training tiles with the selected alpha per target
        x_scaler, y_mean, y_std = fit_scalers(X[tr], y[tr])
        model = Ridge(alpha=best_alphas).fit(x_scaler.transform(X[tr]), (y[tr] - y_mean) / y_std)
        # predictions go back to raw units so folds share one scale
        y_pred[te] = model.predict(x_scaler.transform(X[te])).reshape(len(te), -1) * y_std + y_mean
        fold_of_tile[te] = fold

        for k in range(y.shape[1]):
            fold_records.append({
                "outer_fold": fold,
                "target_index": k,
                "selected_alpha": float(best_alphas[k]),
                "inner_cv_mse": float(inner_mse[:, k].min()),
                "at_grid_edge": bool(best_alphas[k] in (alphas[0], alphas[-1])),
                "n_train_patients": len(np.unique(groups[tr])),
                "n_test_patients": len(np.unique(groups[te])),
                "n_train_tiles": len(tr),
                "n_test_tiles": len(te),
            })

    assert not np.isnan(y_pred).any() and (fold_of_tile >= 0).all(), "a tile was never predicted"
    return y_pred, fold_of_tile, fold_records
