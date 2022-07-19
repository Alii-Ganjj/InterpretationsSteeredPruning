import torch
import torch.nn as nn


class RandomBernoulliSampler(nn.Module):
    def __init__(self, args):
        super(RandomBernoulliSampler, self).__init__()
        self.args = args

    def forward(self, samples):
        n, c, h, w = samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]
        r = torch.rand((n, 1, h, w))
        return (r > 0.5).detach().float().to(self.args.device)
