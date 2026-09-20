"""
/src/hne/models/mlp.py

Distributional MLP for spatial transcriptomics regression.
Accepts any foundation model feature dimension (input_dim) and target count (n_targets).
Predicts parameters of a normal distribution (mu, sigma) per target signature.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class DistributionalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_targets: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_dim: Feature embedding dimension (e.g., 1024 for Phikon, 1280 for Virchow, 512 for CONCH)
            n_targets: Number of continuous signature scores to predict
            hidden_dim: Number of hidden units in intermediate layers
            dropout_rate: Dropout probability for regularizing hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),            # dynamically normalizes incoming FM embeddings
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),                          # prevents dead neurons across negative activations
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # output layer has 2 nodes, one for Mu and one for log_std (considering gene score
            # is normally distributed)
            nn.Linear(hidden_dim, 2 * n_targets),  # 2 outputs per target (signature)
        )

        # output layer initialization:
        # small normal weights ensure backward gradients flow to hidden layers immediately
        last_layer = self.network[-1]
        nn.init.normal_(last_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x: torch.Tensor):
        outputs = self.network(x)

        # split outputs into mu and log_std (each shaped: N, n_targets)
        mu = outputs[:, :self.n_targets]
        log_std = outputs[:, self.n_targets:]

        # bound log_std so sigma stays bounded within [0.05, 2.0]
        # setting bounds for std to prevent exp(log_std) blowing up or hitting zero, bc if the 
        # model starts making bad predictions log_std might drift toward extremely large positive or negative values
        log_std = torch.clamp(log_std, min=-3.0, max=0.7)
        std = torch.exp(log_std)    # std = e^s = exp(log_std) -> making sure SD is strictly positive

        return mu, std  # both (N, n_targets)

    @staticmethod
    def distributional_nll_loss(mu: torch.Tensor, std: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Gaussian Negative Log-Likelihood loss summed over targets and averaged over batch.
        """
        # construct probability density function (normal distribution objects) 
        # with predicted mu and std values
        distro = Normal(loc=mu, scale=std)
        
        # compute log probability of gene score and take negative mean
        # log_prob is (N, n_targets); sum across targets so each tile contributes
        # one scalar loss, then mean over the batch
        return -distro.log_prob(targets).sum(dim=-1).mean()


    
