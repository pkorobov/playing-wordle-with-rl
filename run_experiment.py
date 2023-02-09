import torch
from torch.optim import RMSprop
from argparse import ArgumentParser
import numpy as np
import random

import os

from wordle_rl.agent import RNNAgent, RandomAgent
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


def main(base_seed, total_steps, hid_dim, emb_dim, entropy_coef, logdir):
    nenvs = 6
    nsteps = 32

    assert base_seed >= nenvs

    fix_seed(base_seed)
    env = nature_dqn_env(nenvs=nenvs, seed=[i + base_seed for i in range(nenvs)], logdir=logdir)
    game_voc_matrix = torch.FloatTensor(env.game_voc_matrix)

    _ = env.reset()
    tokenizer = Tokenizer()

    policy = RNNAgent(
        letter_tokens=len(tokenizer.index2letter),
        guess_tokens=len(tokenizer.index2guess_state),
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        num_layers=1,
        output_dim=len(tokenizer.index2letter),
        output_len=5,
        sos_token=1,
        game_voc_matrix=game_voc_matrix
    )

    runner = EnvRunner(env, policy, nsteps=nsteps, transforms=[ComputeValueTargets(policy), MergeTimeBatch()])
    optimizer = RMSprop(policy.parameters(), 7e-4)
    a2c = A2C(policy, optimizer,  entropy_coef=entropy_coef, max_grad_norm=50.0)

    os.makedirs("./model_weights", exist_ok=True)

    env.reset()
    for step in trange(0, total_steps + 1, nenvs * nsteps):
        if step % 1000 == 0 and step > 0:
            torch.save(a2c.policy.state_dict(), f"model_weights/step_{step}")

        a2c.train(runner)
    return a2c


def run_random(base_seed, total_steps, logdir):
    nenvs = 6
    nsteps = 32

    assert base_seed >= nenvs

    fix_seed(base_seed)
    env = nature_dqn_env(nenvs=nenvs, seed=[i + base_seed for i in range(nenvs)], logdir=logdir)
    game_voc_matrix = torch.FloatTensor(env.game_voc_matrix)

    _ = env.reset()
    policy = RandomAgent(game_voc_matrix=game_voc_matrix)
    runner = EnvRunner(env, policy, nsteps=nsteps)

    env.reset()
    for _ in trange(0, total_steps + 1, nenvs * nsteps):
        runner.get_next()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--total_steps', type=int, default=3 * 10 ** 7)
    parser.add_argument('--logdir', type=str, default="wordle")
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--agent', type=str, default="main")
    args = parser.parse_args()

    if args.agent == 'main':
        main(
            base_seed=args.seed,
            total_steps=args.total_steps,
            hid_dim=args.hid_dim,
            emb_dim=args.emb_dim,
            entropy_coef=args.entropy_coef,
            logdir=args.logdir
        )
    else:
        run_random(
            base_seed=args.seed,
            total_steps=args.total_steps,
            logdir=args.logdir
        )
