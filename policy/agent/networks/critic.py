import torch
import torch.nn as nn

import utils


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the Q network
        # increasing the depth makke the results worse
        self.Q = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # TODO: Define the forward pass 
        # Hint: Pass the state and action through the network and return the Q value
        input = torch.cat([obs, action], dim=1)
        q = self.Q(input)
        
        return q

class SacCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # TODO: Define the Q network
        # increasing the depth makke the results worse
        # as suggested in the apper, using two critic network trained parallely to take minimum q value
        self.Q1 = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, 1))

        # self.Q2 = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # TODO: Define the forward pass 
        # Hint: Pass the state and action through the network and return the Q value
        input = torch.cat([obs, action], dim=1)
        q1 = self.Q1(input)
        # q2 = self.Q2(input)
        
        # return torch.minimum(q1, q2)
        return q1