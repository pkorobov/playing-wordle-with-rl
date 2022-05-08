from wordle_rl.wordle_env import WordleEnv
from wordle_rl.wrappers import nature_dqn_env
import numpy as np


def test_deterministic():
    """
    This test checks if seed fixation really works.
    """

    def get_trajectory_words(steps):
        wordle_env = WordleEnv()
        wordle_env.reset(42)
        words = [wordle_env.word]
        for _ in range(steps):
            wordle_env.step(np.array([5, 5, 5, 5, 5]))
            if not np.all(wordle_env.word == words[-1]):
                words.append(wordle_env.word)
        return np.array(words)

    words1 = get_trajectory_words(100)
    words2 = get_trajectory_words(100)

    assert np.all(words1 == words2), f"{words1} != {words2}"
