import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../')
from utils.fft import fft2c, ifft2c

class DataConsistency(nn.Module):
    """
    Data Consistency layer from DC-CNN
    """

    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, x, k0, mask, lambda_):
        A_x = fft2c(x)
        k_dc = (1 - mask) * A_x + mask * (lambda_.squeeze(0) * A_x + (1 - lambda_.squeeze(0)) * k0)
        x_dc = ifft2c(k_dc)
        return x_dc
        