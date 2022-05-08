from wordle_rl.wordle_env import WordleEnv
from wordle_rl.model import RNNAgent
from wordle_rl.wrappers import nature_dqn_env
from wordle_rl.runners import EnvRunner

from wordle_rl.transforms import ComputeValueTargets, MergeTimeBatch
from wordle_rl.a2c import A2C
from tqdm import trange
from wordle_rl.tokenizer import Tokenizer

import numpy as np
import torch
from torch.optim import RMSprop
from run_experiment import main


def test_one_deterministic():
    """
    Test if seed fixation really works.
    """

    def get_trajectory_words(steps):
        wordle_env = WordleEnv()
        wordle_env.reset(42)
        words = [wordle_env.word]
        for _ in range(steps):
            action = np.array([5, 5, 5, 5, 5])
            wordle_env.step(action)
            if not np.all(wordle_env.word == words[-1]):
                words.append(wordle_env.word)
        return np.array(words)

    words1 = get_trajectory_words(100)
    words2 = get_trajectory_words(100)

    assert np.all(words1 == words2), f"{words1} != {words2}"


def test_parallel_deterministic():
    """
    Test if training results are reproducable.
    """
    a2c1 = main(100, 1000)
    a2c2 = main(100, 1000)

    parameters1 = dict(a2c1.policy.named_parameters())
    parameters2 = dict(a2c2.policy.named_parameters())

    for name, value in parameters1.items():
        assert torch.allclose(value, parameters2[name])
