"""
/src/hne/models/evaluation.py

Patient-level evaluation of tile predictions. Every confidence interval and p-value
here treats the patient, not the tile, as the unit of observation.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata

AUC_DEFINITION = (
    "ROC-AUC separating the top third of tiles from the bottom third by true score "
    "(thresholds at the 33rd and 67th percentiles of all evaluated tiles). "
    "The middle third is excluded, so this is easier than an all-tiles AUC."
)


def _safe_r(cov: np.ndarray, var_a: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """Pearson r from sums of products; NaN where either variance is zero."""
    denom = np.sqrt(var_a * var_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 1e-12, cov / denom, np.nan)


def patient_sufficient_stats(y_true: np.ndarray, y_pred: np.ndarray, patient_ids: np.ndarray) -> pd.DataFrame:
    """Per-patient sums needed to rebuild pooled, within- and between-patient r."""
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "y": y_true,
        "p": y_pred,
        "yy": y_true * y_true,
        "pp": y_pred * y_pred,
        "yp": y_true * y_pred,
    })
    stats = df.groupby("patient_id", sort=True).agg(
        n=("y", "size"), sy=("y", "sum"), sp=("p", "sum"),
        syy=("yy", "sum"), spp=("pp", "sum"), syp=("yp", "sum"),
    )
    # sums of squares around each patient's own mean
    stats["cyy"] = stats["syy"] - stats["sy"] ** 2 / stats["n"]
    stats["cpp"] = stats["spp"] - stats["sp"] ** 2 / stats["n"]
    stats["cyp"] = stats["syp"] - stats["sy"] * stats["sp"] / stats["n"]
    stats["mean_true"] = stats["sy"] / stats["n"]
    stats["mean_pred"] = stats["sp"] / stats["n"]
    stats["r_within"] = _safe_r(stats["cyp"].values, stats["cyy"].values, stats["cpp"].values)
    return stats


def _pooled_r(s: Dict[str, np.ndarray]) -> np.ndarray:
    """Pooled r over all tiles of the selected patients. Arrays are (..., n_patients)."""
    n = s["n"].sum(axis=-1)
    sy, sp = s["sy"].sum(axis=-1), s["sp"].sum(axis=-1)
    cov = s["syp"].sum(axis=-1) - sy * sp / n
    var_y = s["syy"].sum(axis=-1) - sy ** 2 / n
    var_p = s["spp"].sum(axis=-1) - sp ** 2 / n
    return _safe_r(cov, var_y, var_p)


def _centered_r(s: Dict[str, np.ndarray]) -> np.ndarray:
    """Pooled r after subtracting each patient's own mean from truth and prediction."""
    return _safe_r(s["cyp"].sum(axis=-1), s["cyy"].sum(axis=-1), s["cpp"].sum(axis=-1))


def _between_r(s: Dict[str, np.ndarray]) -> np.ndarray:
    """r between per-patient mean truth and per-patient mean prediction."""
    a, b = s["mean_true"], s["mean_pred"]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return _safe_r((a * b).sum(axis=-1), (a * a).sum(axis=-1), (b * b).sum(axis=-1))


def _mean_within_r(r_within: np.ndarray) -> np.ndarray:
    return np.nanmean(r_within, axis=-1)


def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def intraclass_correlation(y_true: np.ndarray, patient_ids: np.ndarray) -> Tuple[float, float]:
    """
    One-way random-effects ICC(1) of the true score for unbalanced groups, and the
    effective sample size n_tiles / (1 + (m - 1) * ICC) with m = mean tiles per patient.
    """
    df = pd.DataFrame({"g": patient_ids, "y": y_true})
    grp = df.groupby("g")["y"]
    n_i, mean_i = grp.size().values, grp.mean().values
    n_total, k = len(df), len(n_i)
    grand = df["y"].mean()
    ms_between = (n_i * (mean_i - grand) ** 2).sum() / (k - 1)
    ms_within = ((df["y"] - grp.transform("mean")) ** 2).sum() / (n_total - k)
    n0 = (n_total - (n_i ** 2).sum() / n_total) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    m = n_total / k
    n_eff = n_total / (1 + (m - 1) * max(icc, 0.0))
    return float(icc), float(n_eff)


def evaluate_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    min_tiles_within: int = 5,
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate one signature's predictions.

    Returns:
        metrics: pooled, within-patient, patient-centered and between-patient r, and the
                 high-vs-low AUC, each with a 95% CI from a patient-level bootstrap;
                 patient-level permutation p-values for within- and between-patient r;
                 ICC and effective sample size of the true score.
        per_patient: one row per patient (n_tiles, r_within, mean_true, mean_pred).
    """
    rng = np.random.default_rng(seed)
    stats = patient_sufficient_stats(y_true, y_pred, patient_ids)
    n_patients = len(stats)

    # within-patient r is only defined for patients with enough tiles
    r_within = stats["r_within"].where(stats["n"] >= min_tiles_within).values
    cols = ["n", "sy", "sp", "syy", "spp", "syp", "cyy", "cpp", "cyp", "mean_true", "mean_pred"]
    s = {c: stats[c].values.astype(float) for c in cols}

    # high-vs-low AUC: thresholds fixed on all evaluated tiles
    q_low, q_high = np.quantile(y_true, [1 / 3, 2 / 3])
    keep = (y_true <= q_low) | (y_true >= q_high)
    auc_labels = (y_true[keep] >= q_high).astype(int)
    auc_scores = y_pred[keep]
    auc_patients = pd.Categorical(patient_ids[keep], categories=stats.index).codes
    rows_by_patient = [np.flatnonzero(auc_patients == i) for i in range(n_patients)]

    metrics = {
        "n_patients": n_patients,
        "n_tiles": int(len(y_true)),
        "pooled_r": float(_pooled_r(s)),
        "within_patient_r": float(_mean_within_r(r_within)),
        "n_patients_within": int(np.sum(~np.isnan(r_within))),
        "patient_centered_r": float(_centered_r(s)),
        "between_patient_r": float(_between_r(s)),
        "high_vs_low_auc": _auc_from_scores(auc_labels, auc_scores),
    }

    # === patient-level bootstrap: resample patients with all their tiles
    idx = rng.integers(0, n_patients, size=(n_bootstrap, n_patients))
    sb = {c: v[idx] for c, v in s.items()}
    boot = {
        "pooled_r": _pooled_r(sb),
        "within_patient_r": _mean_within_r(r_within[idx]),
        "patient_centered_r": _centered_r(sb),
        "between_patient_r": _between_r(sb),
        "high_vs_low_auc": np.array([
            _auc_from_scores(auc_labels[rows], auc_scores[rows])
            for rows in (np.concatenate([rows_by_patient[i] for i in draw]) for draw in idx)
        ]),
    }
    for name, values in boot.items():
        metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"] = _percentile_ci(values)

    # === patient-level permutation tests (two-sided)
    # within: under H0 each patient's r is symmetric around 0 -> flip signs per patient
    valid_r = r_within[~np.isnan(r_within)]
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(valid_r)))
    null_within = (signs * valid_r).mean(axis=1)
    metrics["within_patient_p"] = float(
        (1 + np.sum(np.abs(null_within) >= abs(valid_r.mean()))) / (1 + n_permutations)
    )

    # between: under H0 patient mean predictions are exchangeable across patients
    perm = np.argsort(rng.random((n_permutations, n_patients)), axis=1)
    null_between = _between_r({"mean_true": s["mean_true"][None, :], "mean_pred": s["mean_pred"][perm]})
    metrics["between_patient_p"] = float(
        (1 + np.sum(np.abs(null_between) >= abs(metrics["between_patient_r"]))) / (1 + n_permutations)
    )

    metrics["icc_true"], metrics["effective_n_tiles"] = intraclass_correlation(y_true, patient_ids)

    per_patient = stats[["n", "mean_true", "mean_pred"]].rename(columns={"n": "n_tiles"}).copy()
    per_patient["r_within"] = r_within
    return metrics, per_patient.reset_index()
