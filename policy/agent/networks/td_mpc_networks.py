import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

class RewardNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super(RewardNN, self).__init__()

        self.reward = nn.Sequential(
                    nn.Linear(latent_dim+action_dim, hidden_dim), 
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1))

    def forward(self, latent_z, action):
        input = torch.cat([latent_z, action], dim=-1)
        
        return self.reward(input)

class PolicyNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super(PolicyNN, self).__init__()

        self.policy = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim), 
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, action_dim*2))
        self.action_dim = action_dim

    def forward(self, latent_z):
        output = self.policy(latent_z)

        mu, log_std = torch.split(output, [self.action_dim, output.shape[-1] - self.action_dim], dim=-1)

        std = torch.tanh(log_std).exp()+1e-3
        dist = utils.TruncatedNormal(mu, std)
        return dist.sample(clip=1.)

class QNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.Q1 = nn.Sequential( nn.Linear(latent_dim+action_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential( nn.Linear(latent_dim+action_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, latent_z, action):
        input = torch.cat([latent_z, action], dim=-1)
        q1 = self.Q1(input)
        q2 = self.Q2(input)
        
        return torch.minimum(q1, q2)

class EncoderNN(nn.Module):
    def __init__(self, frame_cnt, img_sz, latent_dim, is_image, obs_dim, hidden_dim=128):
        super(EncoderNN, self).__init__()

        if(is_image is False):
            self.enc = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, latent_dim))
            return
        
        C = int(3*frame_cnt)

        layers = [nn.Conv2d(C, 32, 7, stride=2), nn.ReLU(),
				  nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
				  nn.Conv2d(64, 128, 3, stride=2), nn.ReLU()]
        out_shape = self._get_out_shape((C, img_sz, img_sz), layers)
        layers.extend([nn.Flatten(), nn.Linear(np.prod(out_shape), latent_dim)])

        self.enc = nn.Sequential(*layers)

    def _get_out_shape(self, in_shape, layers):
        """Utility function. Returns the output shape of a network for a given input shape."""
        x = torch.randn(*in_shape).unsqueeze(0)
        out_shape = (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape
        return out_shape

    def forward(self, obs):
        x = self.enc(obs)
        
        return x
    
class DynamicsNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super(DynamicsNN, self).__init__()

        self.next_state = nn.Sequential(
                    nn.Linear(latent_dim+action_dim, 128), 
                    nn.ReLU(inplace=True),
                    nn.Linear(128, latent_dim))

    def forward(self, latent_z, action):
        input = torch.cat([latent_z, action], dim=-1)
        
        return self.next_state(input)
