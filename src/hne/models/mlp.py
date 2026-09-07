"""
/scripts/model_train/phase1_mlp/mlp.py

Simple MLP for distributional regression
loss function: negative log-likelihood (NLL)
activation fucntion: ReLU
hidden_dim:
dropout_rate:
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class DistributionalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # output layer has 2 nodes, one for Mu and one for log_std (considering gene score
            # is normally distributed)
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        outputs = self.network(x)
        # split the outputs into 2 vectors
        mu, log_std = torch.chunk(outputs, chunks=2, dim=-1)

        # setting bounds for std to prevent exp(log_std) blowing up or hitting zero, bc if the 
        # model starts making bad predictions log_std might drift toward extremely large positive or negative values
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)    # std = e^s = exp(log_std) -> making sure SD is strictly positive

        return mu, std


    # loss function: Negative Log-Likelihood
    @staticmethod
    def distributional_nll_loss(mu, std, targets):
        """
        Adjust the MLP weights so the predicted parameters maximize the probability
        of ground truth labels
        """
        # construct probability density function (normal distribution objects) 
        # with predicted mu and std values
        distro = Normal(loc=mu, scale=std)

        # compute log probability of gene score and take negative mean
        return -distro.log_prob(targets).mean()
    


