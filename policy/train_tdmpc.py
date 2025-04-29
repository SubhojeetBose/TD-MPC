#!/usr/bin/env python3

import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import (
    ReplayBuffer,
    Episode,
)
from video import TrainVideoRecorder, VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import time
import checkpoint

def make_agent(cfg):
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.cfg.episode_length = self.train_env.spec.max_episode_steps
        
        self.cfg.agent.obs_shape = self.train_env.observation_space.shape
        self.cfg.agent.action_dim = int(self.train_env.action_space.n) # flat action dim
        self.agent = make_agent(self.cfg)

        # create replay buffer
        self.replay_buffer = ReplayBuffer(self.cfg)
        
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
        
        self.eval_env.reset()
        self.video_recorder.init(self.eval_env, enabled=True)
        while eval_until_episode(episode):
            # path = []
            step = 0

            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.plan(self.global_step,
                        obs,
                        1,
                        step==0,
                        eval_mode=True,
                    )
                obs, rew, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                self.video_recorder.record(self.eval_env)
                total_reward += rew
                step += 1

            episode += 1
        self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            # log("success_percentage", np.mean(paths))

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.suite.num_train_frames, self.cfg.suite.action_repeat
        )
        seed_until_step = utils.Until(
            self.cfg.suite.num_seed_frames, self.cfg.suite.action_repeat
        )
        eval_every_step = utils.Every(
            self.cfg.suite.eval_every_frames, self.cfg.suite.action_repeat
        )

        episode_step, episode_reward = 0, 0
        next_snapshot_time = time.time() + self.cfg.snapshot_interval
        next_checkpoint_step = self.global_step + self.cfg.checkpoint_every

        obs, _ = self.train_env.reset()
        done = False

        episode = Episode(self.cfg, obs)

        # self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
        metrics = None
        is_start = True
        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                # if self._global_episode % 10 == 0:
                    # self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # # wait until all the metrics schema is populated
                # observations = np.stack(observations, 0)
                # actions = np.stack(actions, 0)

                self.replay_buffer += episode

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.suite.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", self.replay_buffer.capacity)
                        log("step", self.global_step)

                # reset env
                obs, _ = self.train_env.reset()
                done = False
                episode = Episode(self.cfg, obs) # restart episode
                is_start = True

                # self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
                # try to save snapshot
                if self.cfg.save_snapshot and time.time() > next_snapshot_time:
                    next_snapshot_time = time.time() + self.cfg.snapshot_interval
                    self.save_snapshot(self.cfg.num_snapshots)
                if self.cfg.save_checkpoint and self.global_step > next_checkpoint_step:
                    next_checkpoint_step = self.global_step + self.cfg.checkpoint_every
                    self.save_checkpoint()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.plan(self.global_step,
                        obs,
                        self.cfg.suite.num_seed_frames,
                        is_start,
                        eval_mode=False,
                    )

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                for _ in range(self.cfg.suite.num_iteration_frames):
                    metrics = self.agent.update_model(self.replay_buffer, self.global_step, self.cfg.target_update_freq, self.cfg.tau)
                    self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            obs, rew, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            episode += (obs, action, rew, done) # store in episode later in replay buffer
            episode_reward += rew

            # self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
            episode_step += 1
            self._global_step += 1
            is_start = 0

    def save_snapshot(self, num_snapshots, file_name='snapshot.pt'):
        snapshot = self.work_dir / file_name
        payload = {k: self.__dict__[k] for k in self.keys_to_save}
        payload['agent'] = self.agent.save_snapshot()
        payload['buffer'] = self.replay_buffer.save_snapshot()
        torch.save(payload, snapshot)
        new_file, chksm = checkpoint.save_checkpoint(snapshot, num_snapshots)
        print(f"saved snapshot for step {self.global_step} in {new_file}[{chksm}]")

    def save_checkpoint(self, file_name='checkpoint.pt'):
        self.save_snapshot(-1, file_name)

    def load_checkpoint(self, file_name='checkpoint.pt'):
        self.load_snapshot(file_name)

    @property
    def keys_to_save(self):
        return ["timer", "_global_step", "_global_episode"]

    def load_snapshot(self, file_name='snapshot.pt'):
        snapshot = self.work_dir / file_name
        snapshot = checkpoint.get_checkpoint(snapshot)
        print("load snapshot path", snapshot)
        if snapshot is None or not os.path.isfile(snapshot):
            print("WARNING: snapshot not found")
            return
        payload = torch.load(snapshot, weights_only=False)
        for k in self.keys_to_save:
            self.__dict__[k] = payload[k]
        self.agent.load_snapshot(payload['agent'])
        self.replay_buffer.load_snapshot(payload['buffer'])
        print(f"Loaded snapshot with globalstep {self.global_step}")


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train_tdmpc import Workspace as W

    workspace = W(cfg)
    if cfg.load_checkpoint_path is not None:
        workspace.load_checkpoint()
    elif cfg.save_snapshot:
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
