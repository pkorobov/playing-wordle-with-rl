import torch


DEVICE = torch.device('cpu')


class ComputeValueTargets:
    def __init__(self, policy, gamma=0.75):
        self.policy = policy
        self.gamma = gamma

    def __call__(self, trajectory, latest_observation):
        '''
        This method should modify trajectory inplace by adding
        an item with key 'value_targets' to it

        input:
            trajectory - dict from runner
            latest_observation - last state, numpy, (num_envs x channels x width x height)
        '''
        T = len(trajectory['rewards'])
        targets = [None] * T
        R = self.policy.act(latest_observation)['values']
        for t in range(T - 1, -1, -1):
            rewards = torch.FloatTensor(trajectory['rewards'][t]).to(DEVICE)
            dones = torch.LongTensor(trajectory['dones'][t]).to(DEVICE)
            R = rewards + (1 - dones) * self.gamma * R
            targets[t] = R
        trajectory['value_targets'] = targets


class MergeTimeBatch:
    """ Merges first two axes typically representing time and env batch. """
    def __call__(self, trajectory, latest_observation):
        trajectory['log_probs'] = torch.cat(trajectory['log_probs'], dim=0)
        trajectory['values'] = torch.cat(trajectory['values'], dim=0)
        trajectory['value_targets'] = torch.cat(trajectory['value_targets'], dim=0)
