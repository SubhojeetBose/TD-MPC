import agent.networks.td_mpc_networks as networks
import torch
from copy import deepcopy
import utils
import numpy as np
import torch.nn.functional as F
import random 

class TOLD(nn.Module):
    def __init__(self, frame_cnt, img_sz, latent_dim, action_dim, is_image, obs_dim):
        super().__init__()
        self._encoder = networks.EncoderNN(frame_cnt, img_sz, latent_dim, is_image, obs_dim)
        self._dynamics = networks.DynamicsNN(latent_dim, action_dim)
        self._reward = networks.RewardNN(latent_dim, action_dim)
        self._policy = networks.PolicyNN(latent_dim, action_dim)
        self._q = networks.QNN(latent_dim, action_dim)
    
    def encoder_states(self, obs):
        return self._encoder(obs)

    def next_state_reward(self, latent_z, action):
        return self._dynamics(latent_z, action), self._reward(latent_z, action)
    
    def sample_action(self, latent_z):
        return self._policy(latent_z)

    def get_q(self, latent_z, action):
        return self._q(latent_z, action)

class TDMPC():
    def __init__(self, is_image, obs_dim, frame_cnt, img_sz, latent_dim, action_dim, lr, nums_samples = 500, num_pi_trajs = 50, num_horizon = 5, iterations = 10, rho = 0.5, consistency_coef = 2, reward_coef = 0.5, value_coef = 0.1):
        self.device = torch.device('cuda')
        self.model = TOLD(frame_cnt, img_sz, latent_dim, action_dim, is_image, obs_dim).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=lr)
        self.aug = utils.RandomShiftsAug(img_sz/21)
        self.model.eval()
        self.model_target.eval()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_samples = nums_samples
        self.num_pi_trajs = num_pi_trajs
        self.horizon = num_horizon
        self.iterations = iterations
        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self._prev_mean = None
    
    @torch.no_grad()
    def estimate_value(self, latent_z, actions, horizon, gamma):
        G, discount = 0, 1
        for t in range(horizon):
            next_z, reward = self.model.next_state_reward(latent_z, actions[t])
            G += discount * reward
            discount *= gamma
        G += discount * self.model.get_q(next_z, self.model.sample_action(next_z))
        return G
    

    @torch.no_grad()
    def plan(self, step, obs, num_seed_step, isFirst = True, eval_mode = False):
        # Use random actions for first 5000 steps
        if step < num_seed_step and not eval_mode:
            return torch.empty(self.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)
        
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # sample for N_pi actions using current policy
        pi_actions = torch.empty(self.horizon, self.num_pi_trajs, self.action_dim, device=self.device)
        latent_z = self.model.encoder_states(obs).repeat(self.num_pi_trajs, 1)
        for t in range(self.horizon):
            pi_actions[t] = self.model.sample_action(latent_z)
            latent_z, _ = self.model.next_state_reward(latent_z, pi_actions[t])

        # get the initial latent state and init mean and std for sampling N random shooting actions
        latent_z = self.model.encoder_states(obs).repeat(self.num_samples+self.num_pi_trajs, 1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = 2*torch.ones(self.horizon, self.action_dim, device=self.device)
        if not isFirst and self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]

        # random shooting
        for i in range(self.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.horizon, self.num_samples, self.action_dim, device=std.device), -1, 1)
            actions = torch.cat([actions, pi_actions], dim=1)

            # get top k actions
            value = self.estimate_value(latent_z, actions, self.horizon, 0.99).nan_to_num_(0)
            top_idxs = torch.topk(value.squeeze(1), 64, dim=0).indices
            top_value, top_actions = value[top_idxs], actions[:, top_idxs]

            # get next mean and std deviation
            max_value = top_value.max(0)[0]
            score = torch.exp((top_value - max_value))
            score /= score.sum(0)
            next_mean = torch.sum(score.unsqueeze(0) * top_actions, dim=1) / (score.sum(0) + 1e-9)
            next_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (top_actions - next_mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            next_std = next_std.clamp_(1e-3, 2)
            mean, std = next_mean, next_std

        # get the action based on Jth mean and std
        score = score.squeeze(1).cpu().numpy()
        actions = top_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], next_std[0]
        a = mean
        # if in eval mode we dont random sample else give the mean action
        if not eval_mode:
            a += std * torch.randn(self.action_dim, device=std.device)
        
        return a
    
    def update_pi(self, latent_zs):
        self.pi_optim.zero_grad()

		# Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, latent_z in enumerate(latent_zs):
            a = self.model.sample_action(latent_z)
            Q = torch.min(*self.model.get_q(latent_z, a))
            pi_loss += -Q.mean() * (self.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._policy.parameters(), 10, error_if_nonfinite=False)
        self.pi_optim.step()
        return pi_loss.item()

    @torch.no_grad()
    def get_td_target(self, next_obs, reward):
        next_latent_z = self.model.encoder_states(next_obs)
        td_target = reward + 0.99 * torch.min(*self.model_target.Q(next_latent_z, self.model.sample_action(next_latent_z)))
        
        return td_target

    def update_model(self, replay_buffer, step, target_update_freq, tau):
        obs, action, reward, discount, next_obses = [x.float() for x in next(replay_buffer)]
        start_idx = random.randint(0, len(reward)-self.horizon)

        self.optim.zero_grad()
        self.model.train()

        # get latent_z after running random augmentation
        latent_z = self.model.encoder_states(self.aug(obs[start_idx]))
        latent_zs = [latent_z.detach()]

        # calculating the loss
        consistency_loss, reward_loss, value_loss = 0, 0, 0
        for t in range(start_idx, start_idx+self.horizon):
            q_value = self.model.get_q(latent_z, action[t])
            latent_z, reward_pred = self.model.next(latent_z, action[t])

            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z_target = self.model_target.encoder_states(next_obs)
                td_target = self.get_td_target(next_obs, reward[t])
            latent_zs.append(latent_z.detach())

            # losses
            rho = (self.rho ** t)
            consistency_loss += rho * torch.mean(F.mse_loss(latent_z, next_z_target), dim=1, keepdim=True)
            reward_loss += rho * F.mse_loss(reward_pred, reward[t])
            value_loss += rho * (F.mse_loss(q_value, td_target))

        total_loss = self.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.value_coef * value_loss.clamp(max=1e4)
            
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, error_if_nonfinite=False)
        self.optim.step()

        # update the policy and target model
        pi_loss = self.update_pi(latent_zs)

        # do linear interpolation
        if step % target_update_freq == 0:
            with torch.no_grad():
                for p, p_target in zip(self.model.parameters(), self.model_target.parameters()):
                    p_target.data.lerp_(p.data, tau)
        
        self.model.eval()

        return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}
        










