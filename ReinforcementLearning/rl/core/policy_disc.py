import torch.nn as nn
from ReinforcementLearning.utils.math import *
from ReinforcementLearning.rl.core.distributions import Categorical

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """This function should return a distribution to sample action from"""
        raise NotImplementedError

    def select_action(self, x, mean_action=False):
        dist = self.forward(x)
        action = dist.mean_sample() if mean_action else dist.sample()
        return action

    def get_kl(self, x):
        dist = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist = self.forward(x)
        return dist.log_prob(action)



class PolicyDiscrete(Policy):
    def __init__(self, net, action_num, net_out_dim=None):
        super().__init__()
        self.type = 'discrete'
        net_out_dim = net_out_dim or net.out_dim
        self.net = net
        self.action_head = nn.Linear(net_out_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        action_prob = torch.softmax(self.action_head(x), dim=1)
        return Categorical(probs=action_prob)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}
    


