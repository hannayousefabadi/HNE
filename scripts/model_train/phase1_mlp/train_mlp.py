"""
scripts/model_train/phase1_mlp/train_mlp.py
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

from hne.core.paths import PATIENT_IDS, PHIKON_FEATURES, RESULTS
from hne.models.data import get_cohort_statistics, load_features_and_targets
from hne.models.mlp import DistributionalMLP

FEATURE_REGISTRY = {
    "phikon_v2": {"dir": PHIKON_FEATURES, "suffix": "phikon_features", "dim": 1024},
}

# training Hyperparameters
CONFIG = {
    "output_dir": RESULTS / "mlp_phase1",
    "hidden_dim": 64,
    "dropout_rate": 0.2,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "warmup_epochs": 10,       # Pure MSE loss warm-up for mu before engaging NLL
    "batch_size": 64,
    "val_split": 0.2,
    "patience": 15,
    "seed": 42,
}

# Uses the dynamically calculated cohort z-scores by default
DEFAULT_TARGET_COLS = [
    "FMRP_signature_score_cohort_z",
    "Cell_cycle_signature_score_cohort_z",
    "YAP_signature_score_cohort_z",
    "WNT_signature_score_cohort_z",
    "EMT_signature_score_cohort_z",
]

# command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train distributional MLP on tile features")
    parser.add_argument("--model", choices=FEATURE_REGISTRY.keys(), default="phikon_v2",
                        help="Feature backbone to train on")
    parser.add_argument("--target-cols", nargs="+", default=DEFAULT_TARGET_COLS,
                        help="Signature score columns to model (e.g., Cell_cycle_signature_score_cohort_z)")
    parser.add_argument("--warmup-epochs", type=int, default=CONFIG["warmup_epochs"],
                        help="Number of epochs to pre-train mu with pure MSE")
    return parser.parse_args()


def generate_evaluation_plots(val_df: pd.DataFrame, target_cols: list, output_dir: Path):
    sns.set_theme(style="whitegrid")
    
    for col in target_cols:
        sub = val_df[val_df["target"] == col]
        
        # guard against zero-variance constant predictions during evaluation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConstantInputWarning)
            if np.std(sub["mu_pred"]) > 1e-6 and np.std(sub["y_true"]) > 1e-6:
                r, _ = pearsonr(sub["y_true"], sub["mu_pred"])
            else:
                r = 0.0

        # 1. predicted Mean vs True Target
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

        # 2. calibration Check: Absolute Error vs Predicted Uncertainty (std)
        sub = sub.copy()
        sub["abs_error"] = (sub["y_true"] - sub["mu_pred"]).abs()
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=sub, x="std_pred", y="abs_error", alpha=0.6, color="purple", edgecolor=None)
        plt.title(f"Uncertainty Calibration: {col}")
        plt.xlabel("Predicted Dispersion (σ)")
        plt.ylabel("Absolute Error |y - μ|")
        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{col}.png", dpi=300)
        plt.close()


def plot_loss_curves(metrics_df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss", lw=2)
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Val Loss", lw=2)
    plt.axvline(x=CONFIG["warmup_epochs"], color="gray", linestyle=":", label="Warmup End (NLL start)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE then NLL)")
    plt.title("Distributional MLP Convergence (Two-Stage Warmup)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()


def train():
    args = parse_args()
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_spec = FEATURE_REGISTRY[args.model]

    print(f"\n{'='*60}")
    print(f"Starting Phase 1 MLP Training | Device: {device}")
    print(f"Model: {args.model}")
    print(f"Targets: {args.target_cols}")
    print(f"Warmup: {args.warmup_epochs} epochs with MSE Loss")
    print(f"{'='*60}")

    # 1. patient-level split
    train_patients, val_patients = train_test_split(
        PATIENT_IDS, 
        test_size=CONFIG["val_split"], 
        random_state=CONFIG["seed"]
    )
    print(f"Cohort split: {len(train_patients)} train patients | {len(val_patients)} val patients")

    # 2. leak-free cohort stats calculated from train set only
    z_targets = [col for col in args.target_cols if col.endswith("_cohort_z")]
    cohort_stats = get_cohort_statistics(train_patients, z_targets) if z_targets else None

    # 3. load data
    X_train, y_train, _ = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=train_patients,
        target_cols=args.target_cols,
        filename_suffix=feature_spec["suffix"],
        cohort_stats=cohort_stats,
    )

    X_val, y_val, meta_val = load_features_and_targets(
        features_dir=feature_spec["dir"],
        patient_ids=val_patients,
        target_cols=args.target_cols,
        filename_suffix=feature_spec["suffix"],
        cohort_stats=cohort_stats,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f"Insufficient tiles loaded: {len(X_train)} train, {len(X_val)} val")

    print(f"Tiles loaded: {len(X_train)} train tiles | {len(X_val)} val tiles")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # 4. model setup
    input_dim = X_train.shape[-1]
    n_targets = len(args.target_cols)
    model = DistributionalMLP(
        input_dim=input_dim,
        n_targets=n_targets,
        hidden_dim=CONFIG["hidden_dim"],
        dropout_rate=CONFIG["dropout_rate"]
    ).to(device)

    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # 5. training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print("\nTraining progress:")
    for epoch in range(1, CONFIG["epochs"] + 1):
        is_warmup = epoch <= args.warmup_epochs
        stage_name = "MSE (Warmup)" if is_warmup else "NLL"

        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            mu, std = model(bx)

            if is_warmup:
                loss = mse_loss_fn(mu, by)
            else:
                loss = model.distributional_nll_loss(mu, std, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mu_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                mu, std = model(bx)

                if is_warmup:
                    loss = mse_loss_fn(mu, by)
                else:
                    loss = model.distributional_nll_loss(mu, std, by)

                val_loss += loss.item() * len(bx)
                val_mu_list.append(mu.cpu().numpy())
            val_loss /= len(val_loader.dataset)

        if not is_warmup:
            scheduler.step(val_loss)

        # monitor standard deviation of predictions to ensure variance is present
        all_val_mu = np.concatenate(val_mu_list, axis=0)
        mean_mu_sd = float(np.std(all_val_mu, axis=0).mean())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "stage": stage_name,
            "mean_pred_sd": mean_mu_sd
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == args.warmup_epochs + 1:
            print(f"Epoch {epoch:03d} [{stage_name:<12}] | Train: {train_loss:7.4f} | Val: {val_loss:7.4f} | Pred μ SD: {mean_mu_sd:.4f}")

        # track best loss exclusively after warm-up finishes
        if not is_warmup:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_mlp_distributional.pt")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            # save latest warm-up state so the best weights carry over into NLL stage
            torch.save(model.state_dict(), output_dir / "best_mlp_distributional.pt")

    # 6. save metrics and curves
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)
    plot_loss_curves(metrics_df, output_dir)

    # 7. evaluate best model
    model.load_state_dict(torch.load(output_dir / "best_mlp_distributional.pt"))
    model.eval()

    with torch.no_grad():
        mu_val, std_val = model(torch.from_numpy(X_val).to(device))
        mu_val = mu_val.cpu().numpy()
        std_val = std_val.cpu().numpy()

    val_records = []
    for idx, (patient_id, tile_id) in enumerate(meta_val):
        for target_idx, col in enumerate(args.target_cols):
            val_records.append({
                "patient_id": patient_id,
                "tile_id": tile_id,
                "target": col,
                "y_true": float(y_val[idx, target_idx]),
                "mu_pred": float(mu_val[idx, target_idx]),
                "std_pred": float(std_val[idx, target_idx]),
            })

    predictions_df = pd.DataFrame(val_records)
    predictions_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    generate_evaluation_plots(predictions_df, args.target_cols, output_dir)

    print(f"\nDone. Best Val NLL: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    train()
