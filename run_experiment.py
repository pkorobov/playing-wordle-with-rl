import numpy as np
from wordle_env import WordleEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from model import RNNAgent, get_allowed_letters
from wrappers import nature_dqn_env
from runners import EnvRunner

from transforms import ComputeValueTargets, MergeTimeBatch
from a2c import A2C
from tqdm import trange
from tokenizer import Tokenizer


if __name__ == "__main__":

    nenvs = 4
    nsteps = 32

    env = nature_dqn_env(nenvs=nenvs)
    game_voc_matrix = torch.FloatTensor(env.game_voc_matrix)

    obs = env.reset()

    tokenizer = Tokenizer()

    policy = RNNAgent(
        len(tokenizer.index2letter),
        len(tokenizer.index2guess_state),
        32, 128,
        len(tokenizer.index2letter),
        output_len=5,
        sos_token=1,
        game_voc_matrix=game_voc_matrix
    )

    runner = EnvRunner(env, policy, nsteps=nsteps, transforms=[ComputeValueTargets(policy),
                                                               MergeTimeBatch()])
    optimizer = RMSprop(policy.parameters(), 7e-4)
    a2c = A2C(policy, optimizer, max_grad_norm=1.0)

    total_steps = 10 ** 6

    env.reset()
    for step in trange(0, total_steps + 1, nenvs * nsteps):
        if step % 1000 == 0 and step > 0:
            torch.save(a2c.policy.state_dict(), f"model_weights/step_{step}")

        a2c.train(runner)
