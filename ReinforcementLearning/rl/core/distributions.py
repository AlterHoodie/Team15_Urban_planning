import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical


class Categorical(TorchCategorical):

    def __init__(self, probs=None, logits=None, uniform_prob=0.0):
        super().__init__(probs, logits)
        self.uniform_prob = uniform_prob
        if uniform_prob > 0.0:
            self.uniform = TorchCategorical(logits=torch.zeros_like(self.logits))

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        if self.uniform_prob == 0.0:
            return super().log_prob(value).unsqueeze(1)
        else:
            return super().log_prob(value).unsqueeze(1) * (1 - self.uniform_prob) + self.uniform.log_prob(value).unsqueeze(1) * self.uniform_prob

    def mean_sample(self):
        return self.probs.argmax(dim=1)

    def sample(self):
        if self.uniform_prob == 0.0:
            return super().sample()
        else:
            if torch.bernoulli(torch.tensor(self.uniform_prob)).bool():
                # print('unif')
                return self.uniform.sample()
            else:
                # print('original')
                return super().sample()


