import torch
import torch.nn as nn


""" Helper Functions from RELAX https://github.com/duvenaud/relax """


def safe_log_prob(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps, max=1.0))


def safe_clip(x, eps=1e-8):
    return torch.clamp(x, min=eps, max=1.0)


def gs(x):
    return x.shape


def softplus(x):
    '''
    Let m = max(0, x), then,
    sofplus(x) = log(1 + e(x)) = log(e(0) + e(x)) = log(e(m)(e(-m) + e(x-m)))
                         = m + log(e(-m) + e(x - m))
    The term inside of the log is guaranteed to be between 1 and 2.
    '''
    m = torch.max(torch.zeros(x.shape), x)
    return m + torch.log(torch.exp(-m) + torch.exp(x - m))


def logistic_loglikelihood(z, loc, scale=1):
    return torch.log(torch.exp(-(z - loc) / scale) / scale * torch.square((1 + torch.exp(-(z - loc) / scale))))


def bernoulli_loglikelihood(b, log_alpha):
    return b * (-softplus(-log_alpha)) + (1 - b) * (-log_alpha - softplus(-log_alpha))


def bernoulli_loglikelihood_derivitive(b, log_alpha):
    assert gs(b) == gs(log_alpha)
    sna = torch.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def v_from_u(u, log_alpha, force_same=True, b=None, v_prime=None):
    u_prime = torch.sigmoid(-log_alpha)

    if not force_same:
        v = b * (u_prime + v_prime * (1 - u_prime)) + (1 - b) * v_prime * u_prime
    else:
        v_1 = (u - u_prime) / safe_clip(1 - u_prime)
        v_1 = torch.clamp(v_1, 0, 1)
        v_1 = v_1.detach()
        v_1 = v_1 * (1 - u_prime) + u_prime
        v_0 = u / safe_clip(u_prime)
        v_0 = torch.clamp(v_0, 0, 1)
        v_0 = v_0.detach()
        v_0 = v_0 * u_prime
        v = torch.where((u > u_prime), v_1, v_0)

        if ((torch.isnan(v).sum() != 0) or (torch.isinf(v).sum() != 0)):
            raise ValueError('NaN or Inf in v! v sampling is not numerically stable.')
        if force_same:
            v = v + (-v + u).detach()
    return v


def reparameterize(log_alpha, noise):
    return log_alpha + safe_log_prob(noise) - safe_log_prob(1 - noise)


def concrete_relaxation(z, temp):
    return torch.sigmoid(z / temp)


""" Sampler Classes """


class REBAR_Bernoulli_Sampler(nn.Module):
    """ Layer to Sample z, s, z~ """

    def __init__(self, args):
        super(REBAR_Bernoulli_Sampler, self).__init__()
        self.args = args
        self.tau0 = self.args.tau0

    def forward(self, logits):
        u = torch.rand(logits.shape).to(self.args.device)
        v_p = torch.rand(logits.shape).to(self.args.device)
        z = reparameterize(logits, u)
        s = (z > 0.).float().detach().to(self.args.device)
        v = v_from_u(u, logits, False, s, v_p)
        z_tilde = reparameterize(logits, v)

        sig_z = concrete_relaxation(z, self.tau0)
        sig_z_tilde = concrete_relaxation(z_tilde, self.tau0)

        return [sig_z, s, sig_z_tilde]


class RandomBernoulliSampler(nn.Module):
    def __init__(self, args):
        super(RandomBernoulliSampler, self).__init__()
        self.args = args

    def forward(self, samples):
        n, c, h, w = samples.shape[0], samples.shape[1], samples.shape[2], samples.shape[3]
        r = torch.rand((n, 1, h, w))
        return (r > 0.5).detach().float().to(self.args.device)
