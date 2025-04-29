import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class Episode(object):
    """Storage object for a single episode."""
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.obs_type == 'features' else torch.uint8
        self.obs = torch.zeros((cfg.episode_length+1, *init_obs.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.zeros((cfg.episode_length, cfg.agent.action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((cfg.episode_length,), dtype=torch.float32, device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0
    
    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0
    
    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
        self.action[self._idx] = torch.tensor(action, dtype=self.action.dtype, device=self.action.device)
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


class ReplayBuffer():
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.obs_type == 'features' else torch.uint8
        self.capacity = cfg.replay_buffer_size
        obs_shape = cfg.agent.obs_shape
        self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
        self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.agent.obs_shape), dtype=dtype, device=self.device)
        self._action = torch.empty((self.capacity, cfg.agent.action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx:self.idx+self.cfg.episode_length] = episode.obs[:-1] if self.cfg.obs_type == 'features' else episode.obs[:-1]
        self._last_obs[self.idx//self.cfg.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
        self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.agent.num_horizon
        new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
        new_priorities[mask] = 0
        self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
        self.idx = (self.idx + self.cfg.episode_length) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        if self.cfg.obs_type == 'features':
            return arr[idxs]
        obs = torch.empty((self.cfg.batch_size, 3*self.cfg.suite.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device(self.cfg.device))
        obs = arr[idxs].to(self.cfg.device)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        return obs.float()

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:] if self.cfg.obs_type == 'features' else (3*self.cfg.suite.frame_stack, *self._last_obs.shape[-2:])
        next_obs = torch.empty((self.cfg.agent.num_horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((self.cfg.agent.num_horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.cfg.agent.num_horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
        for t in range(self.cfg.agent.num_horizon+1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs+1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs+1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].to(self.cfg.device).float()
        if not action.is_cuda:
            action, reward, idxs, weights = \
                action.to(self.cfg.device), reward.to(self.cfg.device), idxs.to(self.cfg.device), weights.to(self.cfg.device)

        obs = obs.unsqueeze(1)
        next_obs = next_obs.swapaxes(0, 1)
        action = action.swapaxes(0, 1)
        reward = reward.swapaxes(0, 1)
        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights
    
    @property
    def keys_to_save(self):
        return [ "_obs", "_last_obs", "_action", "_reward", "_priorities", "_eps", "_full", "idx" ]

    def save_snapshot(self):
        return { k: self.__dict__[k] for k in self.keys_to_save }

    def load_snapshot(self, payload):
        for k in self.keys_to_save:
            self.__dict__[k] = payload[k]
