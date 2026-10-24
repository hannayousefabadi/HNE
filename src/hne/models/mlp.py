"""
/src/hne/models/mlp.py

Distributional Multi-Layer Perceptron (MLP) for predicting conditional mean (mu)
and uncertainty (sigma) per pathway signature score.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalMLP(nn.Module):
    """
    Predicts both the expected score (mu) and the uncertainty (sigma) for each target.
    
    Given an input embedding (e.g. 1024-d Phikon-v2), passes through a shared trunk
    with LayerNorm and GELU activations, branching into two heads:
      - mu: unbounded continuous prediction
      - sigma: strictly positive standard deviation via clamped exp(log_sigma)
    """

    def __init__(
        self,
        input_dim: int,
        n_targets: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        min_std: float = 1e-3,
        max_std: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets
        self.min_std = min_std
        self.max_std = max_std

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.mu_head = nn.Linear(hidden_dim, n_targets)
        self.log_std_head = nn.Linear(hidden_dim, n_targets)

        # Initialize heads: mu predicts ~0 initially (for z-scored targets),
        # log_std predicts ~0 so initial sigma is ~1.0
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_std_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)

        # Stable, bounded standard deviation
        # exp(0) = 1.0 (unit variance on z-scored targets)
        std = torch.exp(torch.clamp(log_std, min=-4.0, max=2.5))
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        return mu, std


class GaussianBetaNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood with beta-weighting for stabilized heteroscedastic regression.
    
    Reference: Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks", ICLR 2022.
    
    Standard Gaussian NLL (beta=0):
      L = 0.5 * [(y - mu)^2 / sigma^2 + log(sigma^2)]
      
    Beta-NLL (beta > 0, default 0.5):
      Down-weights the loss of high-uncertainty outliers using (sigma^2)^beta,
      preventing the network from ignoring hard samples by inflating sigma to infinity.
    """

    def __init__(self, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = sigma ** 2
        # Base Gaussian NLL
        sq_err = (target - mu) ** 2
        loss = 0.5 * (sq_err / (var + self.eps) + torch.log(var + self.eps))

        # Beta-weighting factor: detached so gradient is not distorted through the weights
        if self.beta > 0:
            loss = loss * (var.detach() ** self.beta)

        return loss.mean()
