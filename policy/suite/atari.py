from collections import deque
import gymnasium as gym
import numpy as np
import torch
import ale_py
import cv2

gym.register_envs(ale_py)

class Pixels(gym.Wrapper):
    def __init__(self, env, num_frames, height, width):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(num_frames*3,height,width), dtype=np.uint8)
        self.frames = deque([], maxlen=num_frames)
        self.height = height
        self.width = width
    
    def _get_obs(self, is_reset=False):
        # frame transpose: (width, height, color) -> (color, height, width)
        frame = self.env.render()
        frame = cv2.resize(frame, (self.width, self.height)).transpose(2,0,1)
        num_frames = self.frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self.frames.append(frame)
        return np.concatenate(self.frames)
    
    def reset(self, *args, **kwargs):
        _, info = self.env.reset(*args, **kwargs)
        return self._get_obs(is_reset=True), info
    
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(), reward, terminated, truncated, info

class NanCheckWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._check_nan(obs)
        return obs, info

    def step(self, action):
        if np.isnan(action).any():
            raise ValueError("action nan")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._check_nan(obs)
        return obs, reward, terminated, truncated, info

    def _check_nan(self, obs):
        if np.isnan(obs).any():
            raise ValueError("Environment returned observation with NaN values.")

def make(name, seed, num_frames, height, width, obs_type):
    env = gym.make(name, render_mode="rgb_array")
    env = gym.wrappers.RescaleAction(env, -1, 1)
    # env = NanCheckWrapper(env)
    if obs_type == 'pixels':
        env = Pixels(env, num_frames, height, width)
    return env