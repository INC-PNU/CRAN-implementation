
"""Helpful functions for project."""
import os
import re
import torch
from torch.autograd import Variable

from random import shuffle
import numpy as np
import functools
import operator


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def convertTuple(tup):
    str = functools.reduce(operator.add, (tup))
    return str


def to_data(x):
    """Converts variable/tensor to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def spec_to_network_input(x, opts):
    """Converts numpy to variable."""
    freq_size = opts.freq_size
    # trim
    trim_size = freq_size // 2
    # up down 拼接
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if opts.normalization:
        y_abs = torch.abs(y)
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y = torch.div(y, y_abs_max)

    if opts.x_image_channel == 2:
        y = torch.view_as_real(y)  # [B,H,W,2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)  # [B,H,W]
        y = torch.unsqueeze(y, 1)  # [B,H,W]
    return y  # [B,2,H,W]
