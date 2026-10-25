"""
tests/test_abmil.py

Unit tests for GatedAttention, ABMIL architecture, and attention heterogeneity diagnostics.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hne.models.abmil import (
    ABMIL,
    GatedAttention,
    compute_attention_entropy,
    compute_attention_skewness,
    compute_gini_coefficient,
    compute_tile_diagnostics,
    compute_topk_attention_mass,
)


def test_gated_attention():
    print("Testing GatedAttention...")
    m = 75
    d = 1024
    attn_module = GatedAttention(input_dim=d, hidden_dim=128, dropout_rate=0.25)
    h = torch.randn(m, d)

    # Eval mode
    attn_module.eval()
    with torch.no_grad():
        a = attn_module(h)
    assert a.shape == (m, 1), f"Expected shape ({m}, 1), got {a.shape}"
    assert torch.isclose(a.sum(), torch.tensor(1.0), atol=1e-5), f"Expected sum 1.0, got {a.sum()}"
    assert (a > 0).all(), "Attention weights must be positive"

    # Train mode
    attn_module.train()
    a_train = attn_module(h)
    assert a_train.shape == (m, 1)
    assert torch.isclose(a_train.sum(), torch.tensor(1.0), atol=1e-4), f"Expected train sum 1.0, got {a_train.sum()}"
    print("  -> GatedAttention passed.")


def test_abmil_forward():
    print("Testing ABMIL forward pass...")
    d = 1024
    n_targets = 5
    model = ABMIL(input_dim=d, n_targets=n_targets, attention_dim=128, head_hidden_dim=128, dropout_rate=0.25)
    model.eval()

    # Single bag
    bag1 = torch.randn(80, d)
    with torch.no_grad():
        mu, std = model(bag1)
    assert mu.shape == (1, n_targets), f"Expected (1, {n_targets}), got {mu.shape}"
    assert std.shape == (1, n_targets), f"Expected (1, {n_targets}), got {std.shape}"
    assert (std >= 1e-3).all() and (std <= 10.0).all(), "Std outside clamp bounds"

    # List of variable-sized bags (B=3)
    bags = [torch.randn(80, d), torch.randn(55, d), torch.randn(23, d)]
    with torch.no_grad():
        mu, std = model(bags)
    assert mu.shape == (3, n_targets), f"Expected (3, {n_targets}), got {mu.shape}"
    assert std.shape == (3, n_targets), f"Expected (3, {n_targets}), got {std.shape}"

    # List of bags with return_attention=True
    with torch.no_grad():
        mu, std, attns = model(bags, return_attention=True)
    assert len(attns) == 3, f"Expected 3 attention arrays, got {len(attns)}"
    assert attns[0].shape == (80,), f"Expected shape (80,), got {attns[0].shape}"
    assert attns[1].shape == (55,), f"Expected shape (55,), got {attns[1].shape}"
    assert attns[2].shape == (23,), f"Expected shape (23,), got {attns[2].shape}"
    for i, a in enumerate(attns):
        assert np.isclose(a.sum(), 1.0, atol=1e-4), f"Bag {i} attention does not sum to 1: {a.sum()}"
    print("  -> ABMIL forward pass passed.")


def test_heterogeneity_diagnostics():
    print("Testing Intratumoral Heterogeneity diagnostics...")
    # Perfectly uniform weights (high entropy, low gini)
    m = 100
    uniform_weights = np.ones(m) / m
    h_tile, h_norm = compute_attention_entropy(uniform_weights)
    assert np.isclose(h_norm, 1.0, atol=1e-5), f"Uniform norm entropy should be 1.0, got {h_norm}"
    gini = compute_gini_coefficient(uniform_weights)
    assert np.isclose(gini, 0.0, atol=1e-4), f"Uniform Gini should be 0.0, got {gini}"
    top5 = compute_topk_attention_mass(uniform_weights, k=5)
    assert np.isclose(top5, 5 / 100, atol=1e-5), f"Uniform top 5 mass should be 0.05, got {top5}"

    # Spike / hotspot weights (low entropy, high gini)
    spike_weights = np.zeros(m)
    spike_weights[0] = 0.96
    spike_weights[1:] = 0.04 / (m - 1)
    h_tile_s, h_norm_s = compute_attention_entropy(spike_weights)
    assert h_norm_s < 0.2, f"Spike norm entropy should be < 0.2, got {h_norm_s}"
    gini_s = compute_gini_coefficient(spike_weights)
    assert gini_s > 0.9, f"Spike Gini should be > 0.9, got {gini_s}"
    top5_s = compute_topk_attention_mass(spike_weights, k=5)
    assert top5_s > 0.96, f"Spike top 5 mass should be > 0.96, got {top5_s}"

    diag = compute_tile_diagnostics(uniform_weights, top_k=5)
    assert "attention_entropy" in diag
    assert "normalized_attention_entropy" in diag
    assert "gini_coefficient" in diag
    assert "topk_attention_mass" in diag
    assert "attention_skewness" in diag
    assert diag["n_patches"] == 100
    print("  -> Heterogeneity diagnostics passed.")


if __name__ == "__main__":
    test_gated_attention()
    test_abmil_forward()
    test_heterogeneity_diagnostics()
    print("\nALL ABMIL UNIT TESTS PASSED SUCCESSFULLY!")

