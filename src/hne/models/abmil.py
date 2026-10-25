"""
src/hne/models/abmil.py

Attention-Based Multiple Instance Learning (ABMIL) on tile bags for Phase 2.
Implements:
  1. Gated Attention Mechanism (Ilse et al., ICML 2018).
  2. Attention Pooling from variable-sized patch bags (M_i x 1024) to tile embedding (1024-d).
  3. Shared trunk and Distributional Head predicting conditional mean (mu) and spatial uncertainty (sigma).
  4. Intratumoral Heterogeneity (ITH) diagnostics: Shannon Attention Entropy (H_tile),
     Normalized Attention Entropy (H_norm), Gini coefficient, top-k mass, and skewness.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import skew
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """
    Trainable Gated Attention Mechanism (Ilse et al., ICML 2018):
      e_m = w^T [ tanh(V h_m) * sigmoid(U h_m) ]
      a_m = exp(e_m) / sum_j exp(e_j)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.v = nn.Linear(input_dim, hidden_dim)
        self.u = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, return_logits: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            h: Patch embeddings tensor of shape (M, input_dim)
            return_logits: Whether to return unnormalized logits e along with weights a

        Returns:
            a: Attention weights tensor of shape (M, 1), sum(a) == 1
            e (optional): Attention logits tensor of shape (M, 1)
        """
        v_h = torch.tanh(self.v(h))
        u_h = torch.sigmoid(self.u(h))
        e = self.w(v_h * u_h)  # (M, 1)
        a = F.softmax(e, dim=0)  # (M, 1)

        if self.training:
            a = self.dropout(a)
            # Re-normalize across surviving instances so bag embedding remains properly scaled
            a_sum = a.sum()
            if a_sum > 0:
                a = a / a_sum

        if return_logits:
            return a, e
        return a


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) with Distributional Head.

    Aggregates variable-sized patch bags (M_i x input_dim) into a single tile representation
    z_tile via learned gated attention, and predicts (mu, sigma) across 5 pathway signatures.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_targets: int = 5,
        attention_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout_rate: float = 0.25,
        min_std: float = 1e-3,
        max_std: float = 10.0,
        is_distributional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.attention_dim = attention_dim
        self.head_hidden_dim = head_hidden_dim
        self.dropout_rate = dropout_rate
        self.min_std = min_std
        self.max_std = max_std
        self.is_distributional = is_distributional

        # Gated attention pooling module
        self.attention = GatedAttention(
            input_dim=input_dim,
            hidden_dim=attention_dim,
            dropout_rate=dropout_rate,
        )

        # Shared projection trunk on aggregated tile embedding
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Prediction heads
        self.mu_head = nn.Linear(head_hidden_dim, n_targets)
        if is_distributional:
            self.log_std_head = nn.Linear(head_hidden_dim, n_targets)

        # Initialize heads: mu predicts ~0 (for z-scored targets),
        # log_std predicts ~0 so initial sigma is ~1.0
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mu_head.bias)
        if is_distributional:
            nn.init.normal_(self.log_std_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.log_std_head.bias)

    def pool_bag(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pools a single tile bag of patches (M_i, d) into a tile embedding (1, d).
        Returns (z_tile, attention_weights).
        """
        a = self.attention(bag)  # (M_i, 1)
        z = torch.mm(a.t(), bag)  # (1, d)
        return z, a

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        return_attention: bool = False,
    ):
        """
        Args:
            x: Either a List of (M_i, d) bag tensors (from collate_tile_bags)
               or a single 2D tensor (M, d).
            return_attention: If True, returns attention weights list alongside predictions.

        Returns:
            If is_distributional and not return_attention: (mu, std)
            If is_distributional and return_attention: (mu, std, attention_weights)
            If not is_distributional and not return_attention: mu
            If not is_distributional and return_attention: (mu, attention_weights)
        """
        attention_list = []

        if isinstance(x, list):
            pooled = []
            for bag in x:
                z, a = self.pool_bag(bag)
                pooled.append(z)
                if return_attention:
                    attention_list.append(a.squeeze(-1).detach().cpu().numpy())
            z_batch = torch.cat(pooled, dim=0)  # (B, d)
        elif isinstance(x, torch.Tensor):
            if x.ndim == 2:
                z_batch, a = self.pool_bag(x)
                if return_attention:
                    attention_list.append(a.squeeze(-1).detach().cpu().numpy())
            elif x.ndim == 3:
                # Batched fixed-length tensor (B, M, d)
                b, m, d = x.shape
                pooled = []
                for i in range(b):
                    z, a = self.pool_bag(x[i])
                    pooled.append(z)
                    if return_attention:
                        attention_list.append(a.squeeze(-1).detach().cpu().numpy())
                z_batch = torch.cat(pooled, dim=0)
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.ndim}")
        else:
            raise TypeError(f"Unsupported input type for x: {type(x)}")

        features = self.trunk(z_batch)
        mu = self.mu_head(features)

        if self.is_distributional:
            log_std = self.log_std_head(features)
            std = torch.exp(torch.clamp(log_std, min=-4.0, max=2.5))
            std = torch.clamp(std, min=self.min_std, max=self.max_std)
            if return_attention:
                return mu, std, attention_list
            return mu, std
        else:
            if return_attention:
                return mu, attention_list
            return mu


# =========================================================================
# Intratumoral Heterogeneity (ITH) & Attention Distribution Diagnostics
# =========================================================================

def compute_attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> Tuple[float, float]:
    """
    Computes Shannon Attention Entropy (H_tile) and Normalized Entropy (H_norm):
      H_tile = - sum_{m=1}^M a_m * log(a_m + eps)
      H_norm = H_tile / log(M)  in [0, 1]

    High entropy (~1.0): Diffuse attention across all patches (homogeneous tissue).
    Low entropy (<<1.0): Focused attention on small localized hotspots (heterogeneous tissue).
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    m = len(w)
    if m <= 1:
        return 0.0, 1.0

    entropy = -float(np.sum(w * np.log(w + eps)))
    max_entropy = float(np.log(m))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0
    return entropy, float(np.clip(norm_entropy, 0.0, 1.0))


def compute_gini_coefficient(weights: np.ndarray) -> float:
    """
    Computes the Gini coefficient of the attention distribution:
      G = sum_j sum_k |a_j - a_k| / (2 * M * sum_j a_j)

    High Gini (~1.0): Strong spatial inequality / compartmentalization.
    Low Gini (~0.0): Uniform attention across patches.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    m = len(w)
    if m <= 1:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0

    diffs = np.abs(np.subtract.outer(w, w))
    gini = float(diffs.sum() / (2.0 * m * w_sum))
    return float(np.clip(gini, 0.0, 1.0))


def compute_topk_attention_mass(weights: np.ndarray, k: int = 5) -> float:
    """
    Computes the cumulative attention mass concentrated in the top-k patches:
      R_topk = sum_{m in top k} a_m
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    k_eff = min(k, len(w))
    if k_eff == 0:
        return 0.0
    top_weights = np.sort(w)[-k_eff:]
    return float(np.sum(top_weights))


def compute_attention_skewness(weights: np.ndarray) -> float:
    """
    Computes sample skewness of the attention weights distribution.
    Positive skew indicates a long right tail of dominant outlier patches.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if len(w) < 3 or np.std(w) < 1e-9:
        return 0.0
    return float(skew(w))


def compute_tile_diagnostics(weights: np.ndarray, top_k: int = 5) -> Dict[str, float]:
    """
    Aggregates all morphological and attention distribution diagnostics for a single tile.
    """
    h_tile, h_norm = compute_attention_entropy(weights)
    gini = compute_gini_coefficient(weights)
    topk_mass = compute_topk_attention_mass(weights, k=top_k)
    sk = compute_attention_skewness(weights)

    return {
        "n_patches": len(weights),
        "attention_entropy": h_tile,
        "normalized_attention_entropy": h_norm,
        "gini_coefficient": gini,
        "topk_attention_mass": topk_mass,
        "attention_skewness": sk,
    }

