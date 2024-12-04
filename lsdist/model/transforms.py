import torch
from torch import Tensor
import torch.nn as nn


class AddGaussianNoiseTransform(nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self, x: Tensor):
        return x + torch.randn_like(x) * self.std

    def extra_repr(self) -> str:
        return f"std={self.std}"
