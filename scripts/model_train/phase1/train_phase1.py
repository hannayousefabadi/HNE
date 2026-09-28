"""scripts/model_train/phase1_mlp/train_mlp.py"""
import json
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from hne.core.paths import PATIENT_IDS, PHIKON_FEATURES, RESULTS, TILES_SIGNATURE_MATRIX
from hne.models.data import get_cohort_statistics, load_features_and_targets
from hne.models.mlp import DistributionalMLP

FEATURE_REGISTRY = {
    "phikon_v2": {"dir": PHIKON_FEATURES, "suffix": "phikon_features", "dim": 1024},
}

CONFIG = {
    "model_name": "phikon_v2",
    "output_dir": RESULTS / "phase1",
    "hidden_dim": 128,
    "dropout_rate": 0.2,
    "lr": 5e-4,
    "weight_decay": 1e-2,
    "epochs": 80,
    "warmup_epochs": 5,
    "batch_size": 128,
    "val_split": 0.2,
    "patience": 15,
    "seed": 42,
}

DEFAULT_TARGET_COLS = [
    "FMRP_signature_score_cohort_z",
    "Cell_cycle_signature_score_cohort_z",
    "YAP_signature_score_cohort_z",
    "WNT_signature_score_cohort_z",
    "EMT_signature_score_cohort_z",
]

# ==== Helpers ====
def get_available_patient_ids(patient_ids: list[str], feature_spec: dict) -> list[str]:
    """Filter patient IDs to those with preprocessed tiles and extracted features."""
    valid_ids = []
    features_dir = Path(feature_spec["dir"])
    suffix = feature_spec["suffix"]

    for pid in patient_ids:
        tiles_csv = TILES_SIGNATURE_MATRIX / f"tiles_signature_matrix_{pid}.csv"
        if tiles_csv.exists():
            feature_files = list(features_dir.glob(f"{pid}_*_{suffix}.npy"))
            if len(feature_files) > 0:
                valid_ids.append(pid)

    return sorted(valid_ids)


def compute_cohort_pearson(y_true: np.ndarray, mu_pred: np.ndarray) -> tuple[float, list[float]]:
    rs = []
    for k in range(y_true.shape[1]):
        yt, yp = y_true[:, k], mu_pred[:, k]
        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            r, _ = pearsonr(yt, yp)
            rs.append(float(r) if not np.isnan(r) else 0.0)
        else:
            rs.append(0.0)
    return float(np.mean(rs)), rs


def generate_validation_metrics_table(predictions_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Computes and logs comprehensive validation statistics per target signature."""
    records = []
    n_pts = predictions_df["patient_id"].nunique()
    n_tiles = predictions_df["tile_id"].nunique()

    print(f"\n{'='*95}")
    print(f"=== VALIDATION COHORT STATISTICS (N = {n_tiles} tiles, {n_pts} patients) ===")
    print(f"{'='*95}")

    for target in predictions_df["target"].unique():
        sub = predictions_df[predictions_df["target"] == target]
        y_true = sub["y_true"].values
        y_pred = sub["mu_pred"].values

        # Pearson correlation & p-value
        if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            r, p_pearson = pearsonr(y_true, y_pred)
            rho, p_spearman = spearmanr(y_true, y_pred)
        else:
            r, p_pearson, rho, p_spearman = 0.0, 1.0, 0.0, 1.0

        # Binary stratification (Top 33% vs Bottom 33%)
        q_low, q_high = np.quantile(y_true, [0.33, 0.67])
        mask = (y_true <= q_low) | (y_true >= q_high)
        if mask.sum() > 10 and len(np.unique(y_true[mask] >= q_high)) > 1:
            y_binary = (y_true[mask] >= q_high).astype(int)
            auc = roc_auc_score(y_binary, y_pred[mask])
        else:
            auc = float("nan")

        sig = "***" if p_pearson < 0.001 else "**" if p_pearson < 0.01 else "*" if p_pearson < 0.05 else "ns"
        print(f"{target:<38} | r = {r:+.3f} (p={p_pearson:.1e}) {sig:<3} | Spearman rho = {rho:+.3f} | High vs Low AUC = {auc:.3f}")

        records.append({
            "target": target,
            "pearson_r": r,
            "pearson_p_value": p_pearson,
            "spearman_rho": rho,
            "spearman_p_value": p_spearman,
            "high_vs_low_auc": auc,
            "significance": sig,
        })

    print(f"{'='*95}\n")
    metrics_summary_df = pd.DataFrame(records)
    metrics_summary_df.to_csv(output_dir / "validation_metrics_summary.csv", index=False)
    return metrics_summary_df


def generate_evaluation_plots(val_df: pd.DataFrame, target_cols: list, output_dir: Path):
    sns.set_theme(style="whitegrid")
    for col in target_cols:
        sub = val_df[val_df["target"] == col]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            r = (
                pearsonr(sub["y_true"], sub["mu_pred"])[0]
                if (np.std(sub["mu_pred"]) > 1e-6 and np.std(sub["y_true"]) > 1e-6)
                else 0.0
            )

        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=sub, x="y_true", y="mu_pred", alpha=0.6, edgecolor=None)
        y_min = min(sub["y_true"].min(), sub["mu_pred"].min())
        y_max = max(sub["y_true"].max(), sub["mu_pred"].max())
        plt.plot([y_min, y_max], [y_min, y_max], "r--", lw=1.5, label="Identity")
        plt.title(f"{col} (Val Set)\nPearson r = {r:.3f}")
        plt.xlabel("True Signature Score")
        plt.ylabel("Predicted Mean (μ)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pred_vs_true_{col}.png", dpi=300)
        plt.close()


def train():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = CONFIG["model_name"]
    feature_spec = FEATURE_REGISTRY[model_name]

    # === cohort discovery
    available_patients = get_available_patient_ids(PATIENT_IDS, feature_spec)
    print(f"\nCohort discovery: {len(available_patients)} valid preprocessed patients found (out of {len(PATIENT_IDS)} configured).")
    print(f"Starting Phase 1 MLP Training with Pearson-Huber Loss | Device: {device}")

    # === strict patient-level split on verified cohort
    train_patients, val_patients = train_test_split(
        available_patients, test_size=CONFIG["val_split"], random_state=CONFIG["seed"]
    )
    print(f"Cohort split: {len(train_patients)} train | {len(val_patients)} val patients")

    # === calculate leak-free cohort stats
    z_targets = [col for col in DEFAULT_TARGET_COLS if col.endswith("_cohort_z")]
    cohort_stats = get_cohort_statistics(train_patients, z_targets) if z_targets else None

    # === load features and targets
    X_train, y_train, meta_train = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=train_patients,
        target_cols=DEFAULT_TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
        cohort_stats=cohort_stats,
    )

    X_val, y_val, meta_val = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=val_patients,
        target_cols=DEFAULT_TARGET_COLS,
        filename_suffix=feature_spec["suffix"],
        cohort_stats=cohort_stats,
    )

    actual_train_pts = len(set(pid for pid, _ in meta_train))
    actual_val_pts = len(set(pid for pid, _ in meta_val))

    print(f"Tiles loaded: {len(X_train)} train tiles ({actual_train_pts} patients) | "
          f"{len(X_val)} val tiles ({actual_val_pts} patients)")

    # === persist patient split
    split_summary = pd.DataFrame([
        {"patient_id": pid, "split": "train"} for pid in train_patients
    ] + [
        {"patient_id": pid, "split": "val"} for pid in val_patients
    ])
    split_summary.to_csv(output_dir / "patient_cohort_split.csv", index=False)

    # === dataloaders & model setup
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    model = DistributionalMLP(
        input_dim=X_train.shape[-1],
        n_targets=len(DEFAULT_TARGET_COLS),
        hidden_dim=CONFIG["hidden_dim"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)

    best_val_r = -1.0
    patience_counter = 0

    # === taining loop
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            mu, _ = model(bx)

            if epoch <= CONFIG["warmup_epochs"]:
                loss = nn.functional.smooth_l1_loss(mu, by)
            else:
                loss = model.pearson_huber_loss(mu, by, huber_weight=0.5)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            mu_val, _ = model(torch.from_numpy(X_val).to(device))
            mu_val_np = mu_val.cpu().numpy()
            mean_r, r_list = compute_cohort_pearson(y_val, mu_val_np)
            pred_sd = float(np.std(mu_val_np, axis=0).mean())

        if epoch % 5 == 0 or epoch == 1 or epoch == CONFIG["warmup_epochs"] + 1:
            stage = "Warmup" if epoch <= CONFIG["warmup_epochs"] else "PearsonHuber"
            print(f"Epoch {epoch:03d} [{stage:<12}] | Train Loss: {train_loss:.4f} | Val Mean Pearson r: {mean_r:.4f} | Pred SD: {pred_sd:.4f}")

        if epoch > CONFIG["warmup_epochs"]:
            if mean_r > best_val_r:
                best_val_r = mean_r
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_phase1_distributional.pt")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"Early stopping at epoch {epoch}. Best Val Pearson r: {best_val_r:.4f}")
                    break

    # === evaluation of best checkpoint
    model.load_state_dict(torch.load(output_dir / "best_phase1_distributional.pt"))
    model.eval()
    with torch.no_grad():
        mu_val, std_val = model(torch.from_numpy(X_val).to(device))
        mu_val_np = mu_val.cpu().numpy()
        std_val_np = std_val.cpu().numpy()

    val_records = []
    for idx, (patient_id, tile_id) in enumerate(meta_val):
        for target_idx, col in enumerate(DEFAULT_TARGET_COLS):
            val_records.append({
                "patient_id": patient_id,
                "tile_id": tile_id,
                "target": col,
                "y_true": float(y_val[idx, target_idx]),
                "mu_pred": float(mu_val_np[idx, target_idx]),
                "std_pred": float(std_val_np[idx, target_idx]),
            })

    predictions_df = pd.DataFrame(val_records)
    predictions_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    generate_evaluation_plots(predictions_df, DEFAULT_TARGET_COLS, output_dir)
    
    # generate terminal report and save validation_metrics_summary.csv
    generate_validation_metrics_table(predictions_df, output_dir)

    # === save final summary metrics
    summary_info = {
        "model": model_name,
        "n_preprocessed_patients_total": len(available_patients),
        "n_train_patients": actual_train_pts,
        "n_val_patients": actual_val_pts,
        "n_train_tiles": len(X_train),
        "n_val_tiles": len(X_val),
        "best_val_pearson_r": float(best_val_r),
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary_info, f, indent=2)

    print(f"\nTraining Complete. Best Validation Pearson r: {best_val_r:.4f}")
    print(f"Artifacts and summary written to: {output_dir}")


if __name__ == "__main__":
    train()