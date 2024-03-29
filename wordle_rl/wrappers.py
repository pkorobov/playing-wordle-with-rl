""" Environment wrappers. """
from collections import deque

import cv2
import numpy as np

import gym
import gym.spaces as spaces
from wordle_rl.wordle_env import WordleEnv
from tensorboardX import SummaryWriter

from wordle_rl.env_batch import ParallelEnvBatch, WordleParallelEnvBatch


class SequenceWrapper(gym.Wrapper):
    def __init__(self, env, sos_token):
        super(SequenceWrapper, self).__init__(env)
        self.sos_token = sos_token

    def _prepare_obs(self, obs):
        start_obs_part = np.full(shape=(obs.shape[0], obs.shape[1], 1), fill_value=self.sos_token)
        # TODO: use pad token explicitly
        start_obs_part[:, self.num_tries + 1:, :] = 0
        new_obs = np.concatenate([start_obs_part, obs], axis=-1)
        new_obs = new_obs.reshape(obs.shape[0], -1)
        return new_obs
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._prepare_obs(obs)
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._prepare_obs(obs)
        return obs


# it's easier to work without multiprocessing with it
class ReshapeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ReshapeWrapper, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (
            obs[None],
            np.expand_dims(rew, 0),
            np.expand_dims(done, 0),
            [info]
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = obs[None]
        return obs


class TensorboardSummaries(gym.Wrapper):
    """ Writes env summaries."""

    def __init__(
            self, env, prefix=None,
            rew_running_mean_size=100, ep_running_mean_size=5
    ):
        super(TensorboardSummaries, self).__init__(env)
        self.episode_counter = 0
        self.prefix = prefix or self.env.spec.id
        self.writer = SummaryWriter(f"logs/{self.prefix}")
        self.step_var = 0

        self.nenvs = getattr(self.env.unwrapped, "nenvs", 1)
        self.rewards = np.zeros(self.nenvs)
        self.had_ended_episodes = np.zeros(self.nenvs, dtype=bool)
        self.episode_lengths = np.zeros(self.nenvs)
        self.reward_queues = [deque([], maxlen=rew_running_mean_size)
                              for _ in range(self.nenvs)]
        self.episode_length_queues = [deque([], maxlen=ep_running_mean_size)
                                      for _ in range(self.nenvs)]

    def should_write_summaries(self):
        """ Returns true if it's time to write summaries. """
        return np.all(self.had_ended_episodes)

    def add_summaries(self):
        """ Writes summaries. """
        self.writer.add_scalar(
            f"Episodes_length/mean_{self.episode_length_queues[0].maxlen}",
            np.mean([np.mean(q) for q in self.episode_length_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes_length/min_{self.episode_length_queues[0].maxlen}",
            np.min([np.min(q) for q in self.episode_length_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes_length/max_{self.episode_length_queues[0].maxlen}",
            np.max([np.max(q) for q in self.episode_length_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes_reward/total",
            np.mean([q[-1] for q in self.reward_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes_reward/mean_{self.reward_queues[0].maxlen}",
            np.mean([np.mean(q) for q in self.reward_queues]),
            self.step_var
        )
        if self.had_ended_episodes.size > 1:
            self.writer.add_scalar(
                f"Episodes_reward/min_{self.reward_queues[0].maxlen}",
                min(q[-1] for q in self.reward_queues),
                self.step_var
            )
            self.writer.add_scalar(
                f"Episodes_reward/max_{self.reward_queues[0].maxlen}",
                max(q[-1] for q in self.reward_queues),
                self.step_var
            )
        self.had_ended_episodes.fill(False)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards += rew
        self.episode_lengths += 1

        info_collection = [info] if isinstance(info, dict) else info
        done_collection = [done] if isinstance(done, bool) else done
        done_indices = [i for i, info in enumerate(info_collection)
                        if info.get("real_done", done_collection[i])]
        for i in done_indices:
            if not self.had_ended_episodes[i]:
                self.had_ended_episodes[i] = True

            self.reward_queues[i].append(self.rewards[i])
            self.rewards[i] = 0

            self.episode_length_queues[i].append(self.episode_lengths[i])
            self.episode_lengths[i] = 0

        self.step_var += self.nenvs
        if self.should_write_summaries():
            self.add_summaries()
        return obs, rew, done, info

    def reset(self, **kwargs):
        self.rewards.fill(0)
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)
        return self.env.reset(**kwargs)


# magic for parallel launching of environments
class _thunk:
    def __init__(self, i, env_seed, **kwargs):
        self.i = i
        self.env_seed = env_seed
        self.kwargs = kwargs

    def __call__(self):
        return nature_dqn_env(seed=self.env_seed, summaries=False, **self.kwargs)


def nature_dqn_env(nenvs=None, seed=None, summaries=True, logdir="wordle"):
    """ Wraps env as in Nature DQN paper and creates parallel actors. """
    if nenvs is not None:
        if seed is None:
            seed = list(range(nenvs))
        if isinstance(seed, int):
            seed = [seed] * nenvs
        if len(seed) != nenvs:
            raise ValueError(f"seed has length {len(seed)} but must have "
                             f"length equal to nenvs which is {nenvs}")

        thunks = [_thunk(i, env_seed) for i, env_seed in enumerate(seed)]
        env = WordleParallelEnvBatch(thunks)

        if summaries:
            env = TensorboardSummaries(env, prefix=logdir)
        return env

    env = WordleEnv()
    env.reset(seed=seed)
    
    env = SequenceWrapper(env, sos_token=1)

    if summaries:
        env = TensorboardSummaries(env)

    return env
