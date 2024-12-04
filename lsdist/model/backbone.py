import torch
from torch import Tensor
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int],
        output_dim: int,
        activation_module: nn.Module = nn.SiLU,
        dropout: float = 0.5,
    ):
        """A modular Multi-Layer Perceptron (MLP)."""
        super(MLP, self).__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.ouput_dim = output_dim

        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), activation_module())

        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(activation_module())
            hidden_layers.append(nn.Dropout(dropout))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


TBackbone = MLP
