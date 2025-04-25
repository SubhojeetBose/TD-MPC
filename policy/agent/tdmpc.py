import agent.networks.td_mpc_networks as networks
import torch
import torch.nn as nn
from copy import deepcopy
import utils
import numpy as np
import torch.nn.functional as F
import random 

class TOLD(nn.Module):
    def __init__(self, frame_cnt, img_sz, latent_dim, action_dim, is_image, obs_dim):
        super().__init__()
        # self._encoder = networks.EncoderNN(frame_cnt, img_sz, latent_dim, is_image, obs_dim)
        self._dynamics = networks.DynamicsNN(latent_dim, action_dim)
        self._reward = networks.RewardNN(latent_dim, action_dim)
        self._policy = networks.PolicyNN(latent_dim, action_dim)
        self._q = networks.QNN(latent_dim, action_dim)
    
    def encoder_states(self, obs):
        return obs # self._encoder(obs)

    def next_state_reward(self, latent_z, action):
        return self._dynamics(latent_z, action), self._reward(latent_z, action)
    
    def sample_action(self, latent_z):
        return self._policy(latent_z)

    def get_q(self, latent_z, action):
        return self._q(latent_z, action)

class Agent:
    def __init__(self, is_image, obs_dim, frame_cnt, img_sz, latent_dim, action_dim, lr, nums_samples = 500, num_pi_trajs = 50, num_horizon = 5, iterations = 10, rho = 0.5, consistency_coef = 2, reward_coef = 0.5, value_coef = 0.1):
        self.device = torch.device('cpu')
        self.obs_dim = obs_dim
        self.model : TOLD = TOLD(frame_cnt, img_sz, obs_dim[0], action_dim[0], is_image, obs_dim[0]).to(self.device)
        self.model_target = deepcopy(self.model)
        self.q_optim = torch.optim.Adam(self.model._q.parameters(), lr=lr)
        self.dyn_optim = torch.optim.Adam(self.model._dynamics.parameters(), lr=lr)
        self.rew_optim = torch.optim.Adam(self.model._reward.parameters(), lr=lr)
        self.pi_optim = torch.optim.Adam(self.model._policy.parameters(), lr=lr)
        # self.aug = utils.RandomShiftsAug(img_sz/21)
        self.model.eval()
        self.model_target.eval()
        self.action_dim = action_dim[0]
        self.latent_dim = obs_dim[0]
        self.num_samples = nums_samples
        self.num_pi_trajs = num_pi_trajs
        self.horizon = num_horizon
        self.iterations = iterations
        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self._prev_mean = None
        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)
        self.model_target.train(training)
    
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
        # Use random actions for first 500 steps
        if step < num_seed_step and not eval_mode:
            return torch.empty(self.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1).cpu().numpy()
        
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).view(-1, self.latent)
        
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


        return a.cpu().numpy()
    
    def update_pi(self, latent_zs):
        self.pi_optim.zero_grad()

		# Loss is a weighted sum of Q-values
        pi_loss = 0
        for t in range(1):
            a = self.model.sample_action(latent_zs[:, t])
            Q = self.model.get_q(latent_zs[:, t], a)
            pi_loss += -Q.mean() * (self.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._policy.parameters(), 10, error_if_nonfinite=False)
        self.pi_optim.step()
        return pi_loss.item()

    @torch.no_grad()
    def get_td_target(self, next_latent_z, reward):
        td_target = reward + 0.99 * self.model_target.get_q(next_latent_z, self.model.sample_action(next_latent_z)).squeeze(1)
        
        return td_target

    def update_model2(self, replay_buffer, step, target_update_freq, tau):
        obs, action, reward, discount, next_obses = [x.float().to(self.device) for x in next(replay_buffer)]
        
        self.model.train()
        self.model_target.eval()

        obs = obs.view(-1, self.latent_dim)


        # >>> Update Q
        with torch.no_grad():
            next_action = self.model.sample_action(obs)
            target_Q = reward + discount * self.model_target.get_q(next_obses, next_action)

        # TODO: Compute the Q value from the critic network
        Q = self.model.get_q(obs, action)

        # TODO: Compute the critic loss
        critic_loss = F.mse_loss(Q, target_Q)

        # TODO: Optimize the critic network
        # critic gradient descent step
        self.model.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # >>> Pi loss
        latent_z = obs.view(-1, 1, self.latent_dim)
        actor_loss = self.update_pi(latent_z)

        # >>> World loss
        latent_z = obs.detach().clone().view(-1, self.latent_dim)
        pred_obs = self.model._dynamics(latent_z, action)
        dyn_loss = F.mse_loss(pred_obs, next_obses)

        self.dyn_optim.zero_grad()
        dyn_loss.backward()
        self.dyn_optim.step()
        
        # >>> reward model loss
        latent_z = obs.detach().clone().view(-1, self.latent_dim)
        pred_obs = self.model._reward(latent_z, action)
        rew_loss = F.mse_loss(pred_obs, next_obses)

        self.rew_optim.zero_grad()
        rew_loss.backward()
        self.rew_optim.step()
        # <<<< 

        if step % target_update_freq == 0:
            self.soft_update(tau)

        return {'critic_loss': float(critic_loss.item()),
				'actor_loss': actor_loss,
                'dynamics_loss': dyn_loss.item(),
                'reward_loss': rew_loss.item() }


    def update_model(self, replay_buffer, step, target_update_freq, tau):
        obs, action, reward, discount, next_obses = [x.float().to(self.device) for x in next(replay_buffer)]
        start_idx = 0

        self.q_optim.zero_grad()
        self.model.train()

        # get latent_z after running random augmentation
        total_obs = torch.cat((obs[:, start_idx].unsqueeze(1), next_obses[:, start_idx:start_idx+self.horizon]), dim = 1)
        batch_size = total_obs.shape[0]
        latent_nxt_zs = self.model.encoder_states(total_obs.reshape(-1, *total_obs.shape[-3:])).reshape(batch_size, self.horizon+1, -1)
        # print(latent_nxt_zs.shape)
        latent_z = latent_nxt_zs[:, 0].squeeze(1)
        latent_nxt_zs = latent_nxt_zs[:, 1:]
        latent_zs = [latent_z.detach()]

        # calculating the loss
        consistency_loss, reward_loss, value_loss = 0, 0, 0
        for t in range(self.horizon):
            q_value = self.model.get_q(latent_z, action[:, t]).squeeze(1)
            latent_z, reward_pred = self.model.next_state_reward(latent_z, action[:, t])

            with torch.no_grad():
                next_z_target = latent_nxt_zs[:, t].squeeze(1)
                # print(reward[:, t+start_idx].squeeze(1).shape)
                td_target = self.get_td_target(next_z_target, reward[:, t+start_idx].squeeze(1))
            latent_zs.append(latent_z.detach())

            # losses
            rho = (self.rho ** t)
            consistency_loss += rho * torch.mean(F.mse_loss(latent_z, next_z_target), dim=-1, keepdim=True)
            reward_loss += rho * F.mse_loss(reward_pred.squeeze(1), reward[:, t+start_idx].squeeze(1))
            value_loss += rho * (F.mse_loss(q_value, td_target))

        total_loss = self.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.value_coef * value_loss.clamp(max=1e4)
            
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, error_if_nonfinite=False)
        self.q_optim.step()

        # update the policy and target model
        pi_loss = self.update_pi(latent_zs)

        # do linear interpolation
        if step % target_update_freq == 0:
            self.soft_update(tau)
        
        self.model.eval()

        return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'grad_norm': float(grad_norm)}

    def soft_update(self, tau):
        with torch.no_grad():
            for p, p_target in zip(self.model.parameters(), self.model_target.parameters()):
                p_target.data.lerp_(p.data, tau)

    def save_snapshot(self):
        keys_to_save = ["model", "model_target"]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        return payload