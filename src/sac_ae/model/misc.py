import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import numpy as np
import math

def get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    # """Custom weight init for Conv2D and Linear layers."""
    # if isinstance(m, nn.Linear):
    #     #m.weight.data = torch.empty_like(m.weight.data)
    #     nn.init.orthogonal_(m.weight.data)
    #     m.bias.data.fill_(0.0)
    # elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #     # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    #     assert m.weight.size(2) == m.weight.size(3)
    #     m.weight.data.fill_(0.0)
    #     m.bias.data.fill_(0.0)
    #     mid = m.weight.size(2) // 2
    #     gain = nn.init.calculate_gain('relu')
    #     nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class TanhTransform(pyd.transforms.Transform):
    """
    Tanh-transformation class.
    """
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Transformation class to tanh-transform distributions.
    """
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu