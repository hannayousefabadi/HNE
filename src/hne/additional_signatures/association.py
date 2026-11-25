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

from hne.additional_signatures.gene_sets import SINGLE_GENES

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"


def analysis_patients(naive_csv, spot_dir, excluded: dict) -> list[str]:
    """Selected patients that have a spot table. A selected patient without one must be named in `excluded`
    (patient -> reason); any other missing table is an error."""
    selected = pd.read_csv(naive_csv).query("selected")["patient_id"].tolist()
    missing = [p for p in selected if not (spot_dir / f"{p}.csv").exists()]
    unexplained = [p for p in missing if p not in excluded]
    if unexplained:
        raise FileNotFoundError(f"no spot table for {unexplained}: rerun extract_spot_signatures.py or add them to EXCLUDED_PATIENTS")
    for p in missing:
        print(f"excluded {p}: {excluded[p]}")
    return [p for p in selected if p not in missing]


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


def binned_trend(spots: pd.DataFrame, score: str, n_bins: int, n_boot: int, seed: int, bins=None,
                 x: str = "immune_count") -> pd.DataFrame:
    """Mean score per immune-count quantile bin, with a patient-bootstrap CI (patients resampled, then spots pooled).
    `bins` (one bin number per row of spots) replaces the quantile bins when given."""
    bins = pd.qcut(spots[x], n_bins, labels=False, duplicates="drop") if bins is None else bins
    spots = spots.assign(bin=bins)
    patients = spots["patient_id"].unique()
    per_patient = {p: g.groupby("bin")[score].agg(["sum", "count"]) for p, g in spots.groupby("patient_id")}
    bins = sorted(spots["bin"].unique())
    rng = np.random.default_rng(seed)

    def pooled_mean(chosen):
        total = sum(per_patient[p].reindex(bins, fill_value=0) for p in chosen)
        return (total["sum"] / total["count"]).to_numpy()

    boots = np.array([pooled_mean(rng.choice(patients, size=len(patients), replace=True)) for _ in range(n_boot)])
    x_values = spots.groupby("bin")[x].median().reindex(bins).to_numpy()
    return pd.DataFrame({"x": x_values, "mean": pooled_mean(patients),
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


def _p_label(p: float) -> str:
    """A permutation p-value cannot go below 1 / (n_perm + 1), so it is shown as a bound there."""
    return "p < 0.001" if p <= 1e-3 else f"p = {p:.3f}"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_surface(spots, color_score, label, path, cfg, x_score="FMRP_score", x_label="FMRP"):
    """Reference-style figure: colour = color_score over FMRP (x) and immune count (y, log)."""
    xs, ys, surface = smoothed_surface(spots[x_score].to_numpy(), spots["immune_count"].to_numpy(),
                                       spots[color_score].to_numpy(), cfg["surface_grid"],
                                       cfg["surface_bandwidth"], cfg["surface_min_support"])
    fig, ax = plt.subplots(figsize=(6.4, 5))
    mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(surface), cmap="inferno", shading="auto")
    ax.set_yscale("log")
    ax.set_xlabel(f"{x_label} score" if x_score.endswith("_score") else f"{x_label} expression (log1p CPM)", color=INK)
    ax.set_ylabel("Immune cell count (200 µm)", color=INK)
    ax.set_title(f"Association of {x_label} With {label}\nAcross Immune Cell Density", color=INK, fontsize=12)
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
        ax.fill_between(trend["x"], trend["ci_low"], trend["ci_high"], color="#9ecae1", alpha=0.6, linewidth=0)
        ax.plot(trend["x"], trend["mean"], color="#08519c", linewidth=2, marker="o", markersize=5)
        s = summaries[score]
        ax.set_title(score, color=INK, fontsize=11)
        ax.text(0.03, 0.95, f"mean per-patient ρ = {s['mean_rho']:.2f}\n95% CI [{s['ci_low']:.2f}, {s['ci_high']:.2f}]\n"
                f"{_p_label(s['p_perm'])}, n = {s['n_patients']} patients",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Immune cell count (200 µm)", color=INK)
        ax.set_ylabel("expression (log1p CPM)" if score in SINGLE_GENES else "score (ssGSEA)", color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_split_by_detection(spots, score, label, gene_column, gene, path, n_bins, n_boot, seed):
    """`score` against immune count in spots where `gene` has no count vs at least one count.
    Both groups use the same immune-count bins; each band is a patient bootstrap."""
    bins = pd.qcut(spots["immune_count"], n_bins, labels=False, duplicates="drop")
    detected = spots[gene_column] > 0
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mask, name, color in [(~detected, f"{gene} not detected", "#7f7f7f"), (detected, f"{gene} detected", "#08519c")]:
        trend = binned_trend(spots[mask], score, n_bins, n_boot, seed, bins=bins[mask])
        ax.fill_between(trend["x"], trend["ci_low"], trend["ci_high"], color=color, alpha=0.2, linewidth=0)
        ax.plot(trend["x"], trend["mean"], color=color, linewidth=2, marker="o", markersize=5,
                label=f"{name} ({int(mask.sum()):,} spots)")
    ax.set_xscale("log")
    ax.set_xlabel("Immune cell count (200 µm)", color=INK)
    ax.set_ylabel(f"{label} (ssGSEA)", color=INK)
    ax.set_title(f"{label} vs immune count by {gene} detection", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmrp_vs_fmr1(spots, path, n_bins, n_boot, seed, min_spots_per_group):
    """FMRP score against FMR1 expression, for the zero-inflated FMR1 (no count vs at least one count).
    Left: per patient, mean FMRP in spots without vs with an FMR1 count (paired). Right: in spots with a count only,
    FMRP against FMR1 level in quantile bins, patient-bootstrap band. Returns the patient-level statistics."""
    detected = spots["FMR1_expression"] > 0
    paired = {}
    for patient, group in spots.groupby("patient_id"):
        on = group.loc[group["FMR1_expression"] > 0, "FMRP_score"]
        off = group.loc[group["FMR1_expression"] == 0, "FMRP_score"]
        if len(on) >= min_spots_per_group and len(off) >= min_spots_per_group:
            paired[patient] = (off.mean(), on.mean())
    paired = pd.DataFrame(paired, index=["not detected", "detected"]).T
    difference = patient_level_summary(paired["detected"] - paired["not detected"], n_boot, seed)
    rho_all = patient_level_summary(per_patient_rho_pair(spots, "FMRP_score", "FMR1_expression", min_spots_per_group), n_boot, seed)
    rho_detected = patient_level_summary(
        per_patient_rho_pair(spots[detected], "FMRP_score", "FMR1_expression", min_spots_per_group), n_boot, seed)

    fig, (ax_pair, ax_level) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for _, (off, on) in paired.iterrows():
        ax_pair.plot([0, 1], [off, on], color="#bbbbbb", linewidth=1, zorder=1)
    ax_pair.scatter(np.zeros(len(paired)), paired["not detected"], color="#7f7f7f", s=28, zorder=2)
    ax_pair.scatter(np.ones(len(paired)), paired["detected"], color="#08519c", s=28, zorder=2)
    ax_pair.set_xticks([0, 1], ["FMR1 not detected", "FMR1 detected"])
    ax_pair.set_xlim(-0.4, 1.4)
    ax_pair.set_ylabel("mean FMRP score per patient (ssGSEA)", color=INK)
    ax_pair.set_title(f"Per patient (n = {difference['n_patients']})", color=INK, fontsize=11)
    ax_pair.text(0.03, 0.97, f"detected - not detected = {difference['mean_rho']:+.3f}\n"
                 f"95% CI [{difference['ci_low']:+.3f}, {difference['ci_high']:+.3f}]\n{_p_label(difference['p_perm'])}",
                 transform=ax_pair.transAxes, va="top", fontsize=8.5, color=INK)
    _style(ax_pair)

    trend = binned_trend(spots[detected], "FMRP_score", n_bins, n_boot, seed, x="FMR1_expression")
    ax_level.fill_between(trend["x"], trend["ci_low"], trend["ci_high"], color="#9ecae1", alpha=0.6, linewidth=0)
    ax_level.plot(trend["x"], trend["mean"], color="#08519c", linewidth=2, marker="o", markersize=5)
    ax_level.set_xlabel("FMR1 expression, spots with a count (log1p CPM)", color=INK)
    ax_level.set_ylabel("FMRP score (ssGSEA)", color=INK)
    ax_level.set_title("Spots with an FMR1 count", color=INK, fontsize=11)
    ax_level.text(0.03, 0.97, f"mean per-patient ρ = {rho_detected['mean_rho']:.2f}\n95% CI [{rho_detected['ci_low']:.2f}, "
                  f"{rho_detected['ci_high']:.2f}]\n{_p_label(rho_detected['p_perm'])}, n = {rho_detected['n_patients']} patients",
                  transform=ax_level.transAxes, va="top", fontsize=8.5, color=INK)
    _style(ax_level)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"paired_difference": difference, "rho_all_spots": rho_all, "rho_detected_spots": rho_detected}


def per_patient_rho_pair(spots, a: str, b: str, min_spots: int) -> pd.Series:
    """Spearman rho between columns a and b inside each patient; undefined (constant column) or small patients drop out."""
    rho = {}
    for patient, group in spots.groupby("patient_id"):
        if len(group) >= min_spots and group[a].nunique() > 1 and group[b].nunique() > 1:
            rho[patient] = spearmanr(group[a], group[b]).statistic
    return pd.Series(rho, dtype=float)


def plot_updown_opposition(spots, composites: dict, apm_scores: dict, path, min_spots: int, n_boot: int, seed: int):
    """Per patient, rho(FMRP up set, APM) against rho(FMRP down set, APM): one panel per composite (column) and
    APM score (row). A real FMRP-APM link puts patients in an opposite-sign quadrant (shaded): bottom right when
    FMRP goes with APM, top left when against it. composites: name -> (up column, down column);
    apm_scores: label -> column. Returns the per-patient rho table and one summary row per panel, where
    "up_minus_down" is rho(up) - rho(down) per patient (positive = FMRP goes with APM)."""
    fig, axes = plt.subplots(len(apm_scores), len(composites), figsize=(4.2 * len(composites), 4.2 * len(apm_scores)),
                             squeeze=False)
    per_patient, rows = [], []
    for row, (apm_label, apm) in enumerate(apm_scores.items()):
        for col, (name, (up, down)) in enumerate(composites.items()):
            rho = pd.DataFrame({"up": per_patient_rho_pair(spots, up, apm, min_spots),
                                "down": per_patient_rho_pair(spots, down, apm, min_spots)}).dropna()
            rho["up_minus_down"] = rho["up"] - rho["down"]
            per_patient.append(rho.rename_axis("patient_id").reset_index().assign(fmrp=name, apm=apm_label))
            stats = {part: patient_level_summary(rho[part], n_boot, seed) for part in ("up", "down", "up_minus_down")}
            opposite = int(((rho["up"] > 0) & (rho["down"] < 0)).sum() + ((rho["up"] < 0) & (rho["down"] > 0)).sum())
            rows += [{"fmrp": name, "apm": apm_label, "rho_of": part, "n_opposite_sign": opposite, **s}
                     for part, s in stats.items()]

            ax = axes[row, col]
            lim = 1.1 * max(rho[["up", "down"]].abs().max().max(), 0.1)
            ax.fill_between([0, lim], -lim, 0, color="#f0efec", linewidth=0, zorder=0)
            ax.fill_between([-lim, 0], 0, lim, color="#f0efec", linewidth=0, zorder=0)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.plot([-lim, lim], [lim, -lim], color=MUTED, linewidth=0.8, linestyle="--")
            ax.scatter(rho["up"], rho["down"], s=36, color="#2a78d6", edgecolor="white", linewidth=1, zorder=3)
            mu, md = stats["up"], stats["down"]
            ax.errorbar(mu["mean_rho"], md["mean_rho"], xerr=[[mu["mean_rho"] - mu["ci_low"]], [mu["ci_high"] - mu["mean_rho"]]],
                        yerr=[[md["mean_rho"] - md["ci_low"]], [md["ci_high"] - md["mean_rho"]]],
                        fmt="D", color=INK, markersize=7, linewidth=1.5, capsize=3, zorder=4)
            ax.text(0.97, 0.03, f"FMRP with {apm_label}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
            ax.text(0.03, 0.97, f"FMRP against {apm_label}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
            d = stats["up_minus_down"]
            ax.text(0.97, 0.97, f"up {mu['mean_rho']:+.2f}, down {md['mean_rho']:+.2f}\n"
                                f"up - down {d['mean_rho']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}], {_p_label(d['p_perm'])}\n"
                                f"opposite sign: {opposite} of {len(rho)} patients",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            _style(ax)
            ax.set_title(f"FMRP {name.replace('_', ' ')} vs {apm_label}", color=INK, fontsize=11)
            ax.set_xlabel(f"rho(FMRP up genes, {apm_label}) per patient", color=INK)
            ax.set_ylabel(f"rho(FMRP down genes, {apm_label}) per patient", color=INK)
    fig.suptitle("FMRP up and down genes against APM, per patient (diamond: mean, patient-bootstrap 95% CI)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return pd.concat(per_patient, ignore_index=True), pd.DataFrame(rows)
