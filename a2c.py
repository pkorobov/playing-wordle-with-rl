from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
import torch


DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda')


class A2C:
    def __init__(self, policy, optimizer, value_loss_coef=0.25, entropy_coef=0.01, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def loss(self, trajectory, write):
        # compute all losses
        # do not forget to use weights for critic loss and entropy loss

        targets = trajectory['value_targets'].to(DEVICE).detach()
        values = trajectory['values'].to(DEVICE)
        log_probs = trajectory['log_probs'].to(DEVICE)
        value_loss = (targets - values).pow(2).mean()

        # TODO: recompute
        entropy_loss = 0.0  # (log_probs * torch.exp(log_probs)).mean()

        advantage = (targets - values).detach()
        policy_loss = -(log_probs * advantage).mean()

        # log all losses
        write('losses', {
            'policy loss': policy_loss,
            'critic loss': value_loss,
            'entropy loss': entropy_loss
        })

        # additional logs
        write('critic/advantage', advantage.mean())
        write('critic/values', {
            'value predictions': values.mean(),
            'value targets': targets.mean(),
        })

        # return scalar loss
        return policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

    def train(self, runner):
        # collect trajectory using runner
        # compute loss and perform one step of gradient optimization
        # do not forget to clip gradients

        trajectory = runner.get_next()

        self.optimizer.zero_grad()
        loss = self.loss(trajectory, runner.write)
        loss.backward()
        grad_norm = clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        runner.write('gradient norm', grad_norm)
