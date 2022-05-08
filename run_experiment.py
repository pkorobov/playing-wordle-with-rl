import torch
from torch.optim import RMSprop
from argparse import ArgumentParser
import numpy as np
import random


from wordle_rl.model import RNNAgent
from wordle_rl.wrappers import nature_dqn_env
from wordle_rl.runners import EnvRunner

from wordle_rl.transforms import ComputeValueTargets, MergeTimeBatch
from wordle_rl.a2c import A2C
from tqdm import trange
from wordle_rl.tokenizer import Tokenizer


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(base_seed, total_steps):
    nenvs = 6
    nsteps = 32

    assert base_seed >= nenvs

    fix_seed(base_seed)
    env = nature_dqn_env(nenvs=nenvs, seed=[i + base_seed for i in range(nenvs)])
    game_voc_matrix = torch.FloatTensor(env.game_voc_matrix)

    obs = env.reset()

    tokenizer = Tokenizer()

    policy = RNNAgent(
        letter_tokens=len(tokenizer.index2letter),
        guess_tokens=len(tokenizer.index2guess_state),
        emb_dim=16,
        hid_dim=64,
        num_layers=1,
        output_dim=len(tokenizer.index2letter),
        output_len=5,
        sos_token=1,
        game_voc_matrix=game_voc_matrix
    )

    runner = EnvRunner(env, policy, nsteps=nsteps, transforms=[ComputeValueTargets(policy),
                                                               MergeTimeBatch()])
    # optimizer = RMSprop(policy.parameters(), 5e-3)
    optimizer = RMSprop(policy.parameters(), 7e-4)
    a2c = A2C(policy, optimizer,  entropy_coef=0.1, max_grad_norm=50.0)

    env.reset()
    for step in trange(0, total_steps + 1, nenvs * nsteps):
        if step % 1000 == 0 and step > 0:
            torch.save(a2c.policy.state_dict(), f"model_weights/step_{step}")

        a2c.train(runner)
    return a2c


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--total_steps', 10 ** 7, type=int)
    args = parser.parse_args()

    main(base_seed=args.seed, total_steps=args.total_steps)
