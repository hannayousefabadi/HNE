"""src/hne/models/mlp.py"""
import torch
import torch.nn as nn


class DistributionalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_targets: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2 * n_targets),
        )

        last_layer = self.network[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: torch.Tensor):
        outputs = self.network(x)
        mu = outputs[:, :self.n_targets]
        log_std = outputs[:, self.n_targets:]
        # Bounded std strictly between 0.35 and 2.5
        std = torch.exp(torch.clamp(log_std, min=-1.5, max=0.8)) + 0.1
        return mu, std

    @staticmethod
    def pearson_huber_loss(
        mu: torch.Tensor, 
        targets: torch.Tensor, 
        huber_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Differentiable multi-target Pearson Correlation Loss + Huber regularization.
        """
        eps = 1e-6
        total_loss = 0.0
        n_targets = targets.shape[1]

        # Centered variables
        mu_centered = mu - mu.mean(dim=0, keepdim=True)
        targets_centered = targets - targets.mean(dim=0, keepdim=True)

        for k in range(n_targets):
            pred_k = mu_centered[:, k]
            true_k = targets_centered[:, k]

            # Pearson correlation computation
            cov = torch.sum(pred_k * true_k)
            var_pred = torch.sum(pred_k ** 2)
            var_true = torch.sum(true_k ** 2)
            denom = torch.sqrt(var_pred * var_true) + eps

            r = cov / denom
            pearson_loss = 1.0 - r

            # Huber/SmoothL1 component to prevent unbounded scale drift
            huber = nn.functional.smooth_l1_loss(mu[:, k], targets[:, k])

            total_loss += (pearson_loss + huber_weight * huber)

        return total_loss / n_targets