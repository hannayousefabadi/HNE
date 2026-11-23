"""plots/slides/make_interview_figures.py

Slide figures for talks, built only from existing results.

Conventions for these figures:
- "tile" is called "region"; within-patient r is called "within-slide r";
- FMRP is shown as "Signature A" while CONFIG["mask_fmrp"] is True.

Outputs: plots/slides/fig*.png (300 dpi, white background).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hne.core.config import RESULTS
from hne.core.paths import FEATURE_EXTRACTION_QC_REPORTS, PLOTS, PREPROCESSED_COHORT, REGISTRATION_AUDIT, TILES_SIGNATURE_MATRIX

CONFIG = {
    "output_dir": PLOTS / "slides",
    "mask_fmrp": True,
    "signature_order": ["Cell_cycle", "EMT", "WNT", "YAP", "FMRP"],
    "min_regions_within": 5,     # same as the harness (min_tiles_within_patient)
    "n_bootstrap": 2000,
    "seed": 42,
    "n_sigma_bins": 10,
    "dpi": 300,
}

MODELS = {   # label -> (results dir, colour, marker); colours are slots 1-3 of the validated palette
    "Ridge (linear)": (RESULTS / "phase1_ridge", "#2a78d6", "o"),
    "MLP (distributional)": (RESULTS / "phase1b_distributional", "#eb6834", "s"),
    "Attention MIL": (RESULTS / "phase2_abmil", "#1baf7a", "D"),
}

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 11,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})


def label(sig: str) -> str:
    if sig == "FMRP" and CONFIG["mask_fmrp"]:
        return "Signature A"
    return sig.replace("_", " ")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path


def save(fig, name: str):
    out = CONFIG["output_dir"] / name
    fig.savefig(out, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1: cohort
def fig_cohort():
    qc = pd.read_csv(require(PREPROCESSED_COHORT / "qc_summary.csv"))
    n_slides = len(qc)
    n_qc = int((qc["verdict"] != "EXCLUDE").sum())
    n_features = pd.read_csv(require(RESULTS / "phase1_ridge" / "cv_metrics_summary.csv"))["n_patients"].iloc[0]
    n_regions = pd.read_csv(RESULTS / "phase1_ridge" / "cv_metrics_summary.csv")["n_tiles"].iloc[0]

    stages = [
        ("Lung cancer slides\n(H&E + Visium)", n_slides),
        ("Passed automated QC", n_qc),
        ("With image features", n_features),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    y = np.arange(len(stages))[::-1]
    for yi, (name, n) in zip(y, stages):
        ax.barh(yi, n, color=BLUE, height=0.55)
        ax.text(n + 2, yi, f"{n}", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xlim(0, 175)
    ax.set_xlabel("Patients")
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Cohort")
    ax.text(0, -0.36, f"→ {n_regions:,} tumor regions for modelling.   "
            f"{n_slides - n_qc} slides excluded at the tumor-fraction stage.",
            fontsize=10, color=INK2, transform=ax.transAxes)
    save(fig, "fig1_cohort_flow.png")


# ---------------------------------------------------------------- fig 2: registration audit
def fig_registration():
    inv = pd.read_csv(require(REGISTRATION_AUDIT / "slide_inventory.csv"))
    plog = pd.read_csv(require(FEATURE_EXTRACTION_QC_REPORTS / "phikon_v2" / "patient_log.csv"))
    tlog = pd.read_csv(require(FEATURE_EXTRACTION_QC_REPORTS / "phikon_v2" / "tile_log.csv"))

    before_pct = 100 * inv["n_patch_windows_clamped"] / inv["n_patch_windows"].replace(0, np.nan)
    before_pct = before_pct.dropna()
    t = tlog.groupby("patient_id")[["n_out_of_bounds", "n_positions"]].sum()
    after_pct = 100 * t["n_out_of_bounds"] / t["n_positions"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # A: image opened vs coordinate frame of the spots
    after = plog.merge(inv[["patient_id", "implied_fullres_width"]], on="patient_id")
    lim = (1e3, 1.2e5)
    ax1.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax1.text(1.3e3, 1.6e3, "image matches spot coordinates", rotation=37, color=MUTED, fontsize=9)
    ax1.scatter(inv["implied_fullres_width"], inv["slide_width"], s=28, color=ORANGE, edgecolor="white", lw=0.6,
                label=f"Before: instrument preview image (n={len(inv)})")
    ax1.scatter(after["implied_fullres_width"], after["slide_width"], s=28, color=BLUE, marker="s",
                edgecolor="white", lw=0.6, label=f"After: full-resolution scan (n={len(after)})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(*lim); ax1.set_ylim(*lim)
    ax1.set_xlabel("Width of the spot-coordinate frame (px)")
    ax1.set_ylabel("Width of the image opened (px)")
    ax1.set_title("A  Wrong image, silently")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(color=GRID, lw=0.6); ax1.set_axisbelow(True)

    # B: share of image patches that fell outside the image
    rng = np.random.default_rng(CONFIG["seed"])
    for i, (vals, col, mk, name) in enumerate([(before_pct, ORANGE, "o", "Before"), (after_pct, BLUE, "s", "After")]):
        ax2.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=18, color=col, marker=mk, alpha=0.75,
                    edgecolor="white", lw=0.4)
        med = np.median(vals)
        ax2.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
        ax2.text(i + 0.32, med, f"median {med:.0f}%", va="center", fontsize=10, color=INK)
    ax2.set_xticks([0, 1], ["Before", "After"])
    ax2.set_xlim(-0.5, 1.9)
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("Image patches outside the image,\nper patient (%)")
    ax2.set_title("B  Crops clamped to the edge")
    ax2.grid(axis="y", color=GRID, lw=0.6); ax2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig2_registration_audit.png")


# ---------------------------------------------------------------- fig 3: model comparison
def fig_models():
    sigs = CONFIG["signature_order"]
    metrics = {m: pd.read_csv(require(d / "cv_metrics_summary.csv")).set_index("target") for m, (d, _, _) in MODELS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [("within_patient_r", "A  Within-slide spatial correlation"),
              ("between_patient_r", "B  Between-patient correlation")]
    offsets = np.linspace(-0.22, 0.22, len(MODELS))
    y = np.arange(len(sigs))[::-1]
    for ax, (col, title) in zip(axes, panels):
        ax.axvline(0, color=AXIS, lw=1)
        for off, (m, (_, colr, mk)) in zip(offsets, MODELS.items()):
            df = metrics[m].loc[sigs]
            ax.errorbar(df[col], y + off, xerr=[df[col] - df[f"{col}_ci_low"], df[f"{col}_ci_high"] - df[col]],
                        fmt=mk, color=colr, ms=7, mec="white", mew=0.8, elinewidth=1.6, capsize=0, label=m)
        ax.set_title(title)
        ax.set_xlabel("Pearson r (95% patient-bootstrap CI)")
        ax.set_xlim(-0.4, 0.9)
        xgrid(ax)
    axes[0].set_yticks(y, [label(s) for s in sigs])
    axes[0].tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=c, marker=mk, ls="", ms=7, label=m) for m, (_, c, mk) in MODELS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=10.5)
    fig.text(0.01, -0.09, "92 patients, nested cross-validation grouped by patient (5 × 5). "
             "Within-slide p = 1e-4 for every signature and model (patient-level sign-flip permutation).",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_model_comparison.png")


# ---------------------------------------------------------------- fig 4: per-patient r
def fig_per_patient():
    pp = pd.read_csv(require(RESULTS / "phase1_ridge" / "cv_per_patient_metrics.csv"))
    pp = pp[pp["n_tiles"] >= CONFIG["min_regions_within"]].dropna(subset=["r_within"])
    sigs = CONFIG["signature_order"]
    rng = np.random.default_rng(CONFIG["seed"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvline(0, color=AXIS, lw=1)
    for i, s in enumerate(sigs[::-1]):
        v = pp.loc[pp["target"] == s, "r_within"].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=16, color=BLUE, alpha=0.55, edgecolor="white", lw=0.3)
        ax.plot([np.mean(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2.2)
        ax.text(1.04, i, f"{100 * np.mean(v > 0):.0f}% > 0", va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(len(sigs)), [label(s) for s in sigs[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, 1.25)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Within-slide correlation, one dot per patient (Ridge)")
    ax.set_title("Signal holds in most patients, not a few")
    xgrid(ax)
    ax.text(0, -0.17, f"Black bar = mean over {pp['patient_id'].nunique()} patients with ≥ {CONFIG['min_regions_within']} regions.",
            transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, "fig4_per_patient_r.png")


# ---------------------------------------------------------------- fig 5: purity confounder
def _residual(y, x):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def fig_purity():
    preds = pd.read_csv(require(RESULTS / "phase1_ridge" / "cv_predictions.csv"))
    pur = pd.concat([pd.read_csv(f, usecols=["patient_id", "tile_id", "tile_purity"])
                     for f in sorted(TILES_SIGNATURE_MATRIX.glob("tiles_signature_matrix_CH_*.csv"))])
    df = preds.merge(pur, on=["patient_id", "tile_id"], how="left")
    if df["tile_purity"].isna().any():
        raise ValueError(f"{df['tile_purity'].isna().sum()} predictions have no tile purity")

    rows = []
    for (sig, pid), g in df.groupby(["target", "patient_id"]):
        if len(g) < CONFIG["min_regions_within"]:
            continue
        yt, yp, p = g["y_true"].to_numpy(), g["mu_pred"].to_numpy(), g["tile_purity"].to_numpy()
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        raw = np.corrcoef(yt, yp)[0, 1]
        rt, rp = _residual(yt, p), _residual(yp, p)
        partial = np.corrcoef(rt, rp)[0, 1] if np.std(rt) > 0 and np.std(rp) > 0 else np.nan
        rows.append((sig, pid, raw, partial))
    per = pd.DataFrame(rows, columns=["target", "patient_id", "raw", "partial"])

    rng = np.random.default_rng(CONFIG["seed"])
    summary = []
    for sig in CONFIG["signature_order"]:
        g = per[per["target"] == sig]
        out = {"target": sig}
        for col in ("raw", "partial"):
            v = g[col].dropna().to_numpy()
            boots = [rng.choice(v, len(v)).mean() for _ in range(CONFIG["n_bootstrap"])]
            out[col], out[f"{col}_lo"], out[f"{col}_hi"] = v.mean(), *np.percentile(boots, [2.5, 97.5])
        summary.append(out)
    s = pd.DataFrame(summary).set_index("target")
    s.to_csv(CONFIG["output_dir"] / "fig5_purity_partial_r.csv")

    sigs = CONFIG["signature_order"]
    y = np.arange(len(sigs))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for off, col, colr, mk, name in [(0.13, "raw", BLUE, "o", "As reported"),
                                     (-0.13, "partial", ORANGE, "s", "Tumor purity removed")]:
        d = s.loc[sigs]
        ax.errorbar(d[col], y + off, xerr=[d[col] - d[f"{col}_lo"], d[f"{col}_hi"] - d[col]], fmt=mk, color=colr,
                    ms=7, mec="white", mew=0.8, elinewidth=1.6, label=name)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [label(x) for x in sigs])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.05, 0.55)
    ax.set_xlabel("Mean within-slide r (95% patient-bootstrap CI), Ridge")
    ax.set_title("Not just tumor cellularity")
    ax.legend(loc="lower left", fontsize=10)
    xgrid(ax)
    ax.text(0, -0.2, "Partial r: region tumor purity regressed out of truth and prediction inside each patient. "
            "Exploratory (post hoc).", transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, "fig5_purity_confounder.png")
    print(s.round(3))


# ---------------------------------------------------------------- fig 6: uncertainty
def fig_uncertainty():
    preds = pd.read_csv(require(RESULTS / "phase2_abmil" / "cv_predictions.csv"))
    cal = pd.read_csv(require(RESULTS / "phase2_abmil" / "calibration_summary.csv")).set_index("target")
    sigs = CONFIG["signature_order"]
    fig, axes = plt.subplots(1, len(sigs), figsize=(13, 3.4))
    for ax, sig in zip(axes, sigs):
        g = preds[preds["target"] == sig].copy()
        g["abs_err"] = (g["y_true"] - g["mu_pred"]).abs()
        g["bin"] = pd.qcut(g["std_pred"], CONFIG["n_sigma_bins"], labels=False, duplicates="drop")
        b = g.groupby("bin").agg(sigma=("std_pred", "mean"), err=("abs_err", "mean"))
        hi = max(b["sigma"].max(), b["err"].max()) * 1.15
        ax.plot([0, hi], [0, hi * np.sqrt(2 / np.pi)], color=MUTED, ls="--", lw=1)
        ax.plot(b["sigma"], b["err"], color="#1baf7a", lw=2, marker="o", ms=5, mec="white")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(label(sig), fontsize=11.5)
        ax.text(0.04, 0.93, f"ρ = {cal.loc[sig, 'error_vs_std_spearman']:+.2f}", transform=ax.transAxes, fontsize=10)
        ax.text(0.04, 0.82, f"±1σ cover {cal.loc[sig, 'coverage_1sigma_pct']:.0f}%", transform=ax.transAxes,
                fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Mean |error| in bin")
    fig.supxlabel("Predicted σ (deciles of regions)", fontsize=11, color=INK2, y=0.02)
    fig.suptitle("Predicted uncertainty ranks real error (Attention MIL)", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, -0.08, "Dashed: perfectly calibrated Gaussian (mean |error| = 0.8σ). ρ = Spearman(|error|, σ) over all regions, "
             "p < 1e-16 for all. Nominal ±1σ coverage is 68%: intervals are too narrow.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig6_uncertainty_calibration.png")


# ---------------------------------------------------------------- fig 7: effective sample size
def fig_effective_n():
    m = pd.read_csv(require(RESULTS / "phase1_ridge" / "cv_metrics_summary.csv")).set_index("target")
    sigs = CONFIG["signature_order"]
    n_regions = int(m["n_tiles"].iloc[0])
    n_patients = int(m["n_patients"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = ["All regions"] + [label(s) for s in sigs] + ["Patients"]
    vals = [n_regions] + [m.loc[s, "effective_n_tiles"] for s in sigs] + [n_patients]
    y = np.arange(len(names))[::-1]
    colors = [MUTED] + [BLUE] * len(sigs) + [MUTED]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v, s in zip(y, vals, [None] + sigs + [None]):
        txt = f"{v:,.0f}" if s is None else f"{v:,.0f}   (ICC {m.loc[s, 'icc_true']:.2f})"
        ax.text(v + 30, yi, txt, va="center", fontsize=10)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, n_regions * 1.25)
    ax.set_xlabel("Number of samples")
    ax.set_title("Regions are not independent samples")
    xgrid(ax)
    ax.text(0, -0.27, "Blue: effective sample size n / (1 + (m − 1)·ICC). Hence: folds, CIs and p-values all count patients.",
            transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, "fig7_effective_sample_size.png")


if __name__ == "__main__":
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    fig_cohort()
    fig_registration()
    fig_models()
    fig_per_patient()
    fig_purity()
    fig_uncertainty()
    fig_effective_n()
