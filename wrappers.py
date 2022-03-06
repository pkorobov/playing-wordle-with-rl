""" Environment wrappers. """
from collections import deque

import cv2
import numpy as np

import gym
import gym.spaces as spaces
from wordle_env import WordleEnv
from gym.wrappers import Monitor
from tensorboardX import SummaryWriter

from env_batch import ParallelEnvBatch

cv2.ocl.setUseOpenCL(False)


class StartWithRandomActions(gym.Wrapper):
    """ Makes random number of random actions at the beginning of each
    episode. """

    def __init__(self, env, max_random_actions=30):
        super(StartWithRandomActions, self).__init__(env)
        self.max_random_actions = max_random_actions
        self.real_done = True

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.real_done = info.get("real_done", True)
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        if self.real_done:
            num_random_actions = np.random.randint(self.max_random_actions + 1)
            for _ in range(num_random_actions):
                obs, _, _, _ = self.env.step(self.env.action_space.sample())
            self.real_done = False
        return obs


class TokenizerWrapper(gym.Wrapper):
    def __init__(self, env, tokenizer):
        super(TokenizerWrapper, self).__init__(env)
        self.tokenizer = tokenizer




class ClipReward(gym.RewardWrapper):
    """ Modifes reward to be in {-1, 0, 1} by taking sign of it. """

    def reward(self, reward):
        return np.sign(reward)


class TensorboardSummaries(gym.Wrapper):
    """ Writes env summaries."""

    def __init__(self, env, prefix=None, running_mean_size=100, step_var=None):
        super(TensorboardSummaries, self).__init__(env)
        self.episode_counter = 0
        self.prefix = prefix or self.env.spec.id
        self.writer = SummaryWriter(f"logs/{self.prefix}")
        self.step_var = 0

        self.nenvs = getattr(self.env.unwrapped, "nenvs", 1)
        self.rewards = np.zeros(self.nenvs)
        self.had_ended_episodes = np.zeros(self.nenvs, dtype=np.bool)
        self.episode_lengths = np.zeros(self.nenvs)
        self.reward_queues = [deque([], maxlen=running_mean_size)
                              for _ in range(self.nenvs)]

    def should_write_summaries(self):
        """ Returns true if it's time to write summaries. """
        return np.all(self.had_ended_episodes)

    def add_summaries(self):
        """ Writes summaries. """
        self.writer.add_scalar(
            f"Episodes/total_reward",
            np.mean([q[-1] for q in self.reward_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes/reward_mean_{self.reward_queues[0].maxlen}",
            np.mean([np.mean(q) for q in self.reward_queues]),
            self.step_var
        )
        self.writer.add_scalar(
            f"Episodes/episode_length",
            np.mean(self.episode_lengths),
            self.step_var
        )
        if self.had_ended_episodes.size > 1:
            self.writer.add_scalar(
                f"Episodes/min_reward",
                min(q[-1] for q in self.reward_queues),
                self.step_var
            )
            self.writer.add_scalar(
                f"Episodes/max_reward",
                max(q[-1] for q in self.reward_queues),
                self.step_var
            )
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards += rew
        self.episode_lengths[~self.had_ended_episodes] += 1

        info_collection = [info] if isinstance(info, dict) else info
        done_collection = [done] if isinstance(done, bool) else done
        done_indices = [i for i, info in enumerate(info_collection)
                        if info.get("real_done", done_collection[i])]
        for i in done_indices:
            if not self.had_ended_episodes[i]:
                self.had_ended_episodes[i] = True
            self.reward_queues[i].append(self.rewards[i])
            self.rewards[i] = 0

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
        return nature_dqn_env(seed=self.env_seed, summaries=False, clip_reward=False, **self.kwargs)


def nature_dqn_env(nenvs=None, seed=None, summaries=True, monitor=False, clip_reward=True):
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
        env = ParallelEnvBatch(thunks)

        if summaries:
            env = TensorboardSummaries(env, prefix="wordle")
        if clip_reward:
            env = ClipReward(env)
        return env

    env = WordleEnv()
    env.seed(seed)

    if summaries:
        env = TensorboardSummaries(env)

    env = StartWithRandomActions(env, max_random_actions=30)

    if monitor:
        env = gym.wrappers.Monitor(env, directory="videos", force=True)

    if clip_reward:
        env = ClipReward(env)
    return env