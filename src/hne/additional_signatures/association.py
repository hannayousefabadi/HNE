"""src/hne/additional_signatures/association.py

Statistics and plots for the association of FMRP / APM scores with immune count in cancer-dominant spots.
Spots are only used to draw the pictures; every interval and p-value counts patients
(patient-level bootstrap, patient-level sign-flip permutation of the per-patient Spearman rho).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"


def load_spot_tables(spot_dir, patient_ids) -> pd.DataFrame:
    missing = [p for p in patient_ids if not (spot_dir / f"{p}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"no spot table for {missing}, run extract_spot_signatures.py")
    return pd.concat([pd.read_csv(spot_dir / f"{p}.csv") for p in patient_ids], ignore_index=True)


def per_patient_rho(spots: pd.DataFrame, score: str, min_spots: int) -> pd.Series:
    """Spearman rho between score and immune count inside each patient; patients with too few spots drop out."""
    rho = {}
    for patient, group in spots.groupby("patient_id"):
        if len(group) >= min_spots:
            rho[patient] = spearmanr(group[score], group["immune_count"]).statistic
    return pd.Series(rho, name=score)


def patient_level_summary(rho: pd.Series, n_boot: int, seed: int) -> dict:
    """Mean per-patient rho, patient-bootstrap 95% CI, two-sided sign-flip permutation p-value."""
    rng = np.random.default_rng(seed)
    values = rho.dropna().to_numpy()
    boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    flips = rng.choice([-1.0, 1.0], size=(n_boot, len(values)))
    null = (flips * values).mean(axis=1)
    return {"n_patients": len(values), "mean_rho": values.mean(),
            "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "p_perm": (np.sum(np.abs(null) >= abs(values.mean())) + 1) / (n_boot + 1)}


def binned_trend(spots: pd.DataFrame, score: str, n_bins: int, n_boot: int, seed: int) -> pd.DataFrame:
    """Mean score per immune-count quantile bin, with a patient-bootstrap CI (patients resampled, then spots pooled)."""
    spots = spots.assign(bin=pd.qcut(spots["immune_count"], n_bins, labels=False, duplicates="drop"))
    patients = spots["patient_id"].unique()
    per_patient = {p: g.groupby("bin")[score].agg(["sum", "count"]) for p, g in spots.groupby("patient_id")}
    bins = sorted(spots["bin"].unique())
    rng = np.random.default_rng(seed)

    def pooled_mean(chosen):
        total = sum(per_patient[p].reindex(bins, fill_value=0) for p in chosen)
        return (total["sum"] / total["count"]).to_numpy()

    boots = np.array([pooled_mean(rng.choice(patients, size=len(patients), replace=True)) for _ in range(n_boot)])
    x = spots.groupby("bin")["immune_count"].median().reindex(bins).to_numpy()
    return pd.DataFrame({"immune_count": x, "mean": pooled_mean(patients),
                         "ci_low": np.nanpercentile(boots, 2.5, axis=0),
                         "ci_high": np.nanpercentile(boots, 97.5, axis=0)})


def smoothed_surface(x, y, z, grid_size: int, bandwidth: float, min_support: float):
    """Gaussian-kernel mean of z over (x, log10 y). Cells with kernel mass below min_support spots are masked."""
    ly = np.log10(y)
    xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), grid_size)
    ys = np.linspace(np.percentile(ly, 1), np.percentile(ly, 99), grid_size)
    sx, sy = bandwidth * (xs[-1] - xs[0]), bandwidth * (ys[-1] - ys[0])
    gx, gy = np.meshgrid(xs, ys)
    num, den = np.zeros_like(gx), np.zeros_like(gx)
    for xi, yi, zi in zip(x, ly, z):
        w = np.exp(-0.5 * (((gx - xi) / sx) ** 2 + ((gy - yi) / sy) ** 2))
        num += w * zi
        den += w
    surface = np.where(den >= min_support, num / np.maximum(den, 1e-12), np.nan)
    return xs, 10 ** ys, surface


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_surface(spots, color_score, label, path, cfg):
    """Reference-style figure: colour = color_score over FMRP (x) and immune count (y, log)."""
    xs, ys, surface = smoothed_surface(spots["FMRP_score"].to_numpy(), spots["immune_count"].to_numpy(),
                                       spots[color_score].to_numpy(), cfg["surface_grid"],
                                       cfg["surface_bandwidth"], cfg["surface_min_support"])
    fig, ax = plt.subplots(figsize=(6.4, 5))
    mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(surface), cmap="inferno", shading="auto")
    ax.set_yscale("log")
    ax.set_xlabel("FMRP score", color=INK)
    ax.set_ylabel("Immune cell count (200 µm)", color=INK)
    ax.set_title(f"Association of FMRP Signature With {label}\nAcross Immune Cell Density", color=INK, fontsize=12)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"{label}", color=INK)
    cbar.ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_trends(trends: dict, summaries: dict, path):
    """One panel per score: binned mean vs immune count with patient-bootstrap band; rho annotated."""
    fig, axes = plt.subplots(1, len(trends), figsize=(4.2 * len(trends), 3.8), sharex=True)
    for ax, (score, trend) in zip(np.atleast_1d(axes), trends.items()):
        ax.fill_between(trend["immune_count"], trend["ci_low"], trend["ci_high"], color="#9ecae1", alpha=0.6, linewidth=0)
        ax.plot(trend["immune_count"], trend["mean"], color="#08519c", linewidth=2, marker="o", markersize=5)
        s = summaries[score]
        ax.set_title(score, color=INK, fontsize=11)
        ax.text(0.03, 0.95, f"mean per-patient ρ = {s['mean_rho']:.2f}\n95% CI [{s['ci_low']:.2f}, {s['ci_high']:.2f}]\n"
                f"p = {s['p_perm']:.3f}, n = {s['n_patients']} patients",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Immune cell count (200 µm)", color=INK)
        ax.set_ylabel("score (ssGSEA)", color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
