import torch
from torch import nn

import utils

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the policy network
        self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        # TODO: Define the forward pass
        mu = self.policy(obs)

        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class SacActor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the policy network
        self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]*2))
        self.action_shape = action_shape[0]

        self.apply(utils.weight_init)

    def forward(self, obs):
        # TODO: Define the forward pass
        output = self.policy(obs)

        # calculate std
        mu, log_std = torch.split(output, [self.action_shape, output.shape[-1] - self.action_shape], dim=-1)
        # needed to fix the range of std, otherwise it reults get too low for gradient descent
        std = torch.tanh(log_std).exp()+1e-3
        dist = utils.TruncatedNormal(mu, std)
        return dist.sample(clip=0.3)
