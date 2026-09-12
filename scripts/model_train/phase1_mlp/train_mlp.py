"""
/scripts/model_train/phase1_mlp/train_mlp.py
"""
import argparse
import numpy as np
import torch
from pathlib import Path


from hne.models.mlp import DistributionalMLP
from hne.models.data import load_features_and_targets
from hne.core.paths import PHIKON_FEATURES, PATIENT_IDS

FEATURE_REGISTRY = {
    "phikon_v2": {"dir": PHIKON_FEATURES, "suffix": "phikon_features"},
}


def parse_args():
    """CLI arguments"""
    parser = argparse.ArgumentParser(description="Train distributional MLP on tile features")
    parser.add_argument("--model", choices=FEATURE_REGISTRY.keys(), default="phikon_v2",
                        help="Which feature set to train on")
    parser.add_argument("--target-col", type=str, required=True, 
                        help="Signature score columns to predict")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def check_feature_dim(feature_dir, expected_dim=None):
    sample = next(Path(feature_dir).glob("*.npy"))
    dim = np.load(sample).shape[-1]
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Feature dim mismatch: found {dim}, expected {expected_dim} in {feature_dir}")
    return dim


def train():    
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train, y_train = load_features_and_targets()
    input_dim = X_train.shape[-1]
    model = DistributionalMLP(input_dim=input_dim)

    mu, std = model.forward(X_train)
    loss = model.distributional_nll_loss(mu=mu, std=std, targets=y_train)



if __name__ == "__main__":
    train()    
    