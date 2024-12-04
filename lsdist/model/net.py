from functools import cache
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer, AdamW

from ..llama.model import RMSNorm
from ..utils import get_env
from .backbone import TBackbone

DEV = get_env() == "development"
DEFAULT_KDE_SIGMA = 32.0
DEFAULT_LOSS_TYPE = "kld"


class DiscreteDistributionNet(nn.Module):
    def __init__(
        self,
        backbone: TBackbone,
        max_seq_len: int,
        norm: bool = False,
        norm_module: nn.Module = RMSNorm,
        activation: bool = True,
        activation_module: nn.Module = nn.SiLU,
    ):
        super().__init__()

        layers = [
            norm_module(backbone.input_dim) if norm else None,
            backbone,
            activation_module() if activation else None,
            nn.Linear(backbone.ouput_dim, max_seq_len + 1),
        ]

        self.layers = nn.Sequential(*filter(lambda l: l is not None, layers))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

    @staticmethod
    def loss_fn(
        input: Tensor,
        target: Tensor,
        kde_sigma: float = DEFAULT_KDE_SIGMA,
        loss_type: str = DEFAULT_LOSS_TYPE,
    ):
        input_pdf = F.softmax(input, dim=1)
        target_pdf = kde(target, sigma=kde_sigma)

        if loss_type == "mse":
            loss = F.mse_loss(input_pdf, target_pdf, reduction="sum")
        elif loss_type == "kld":
            loss = F.kl_div(input_pdf.log(), target_pdf, reduction="sum")
        else:
            raise ValueError("Unsupported loss type. Choose 'mse' or 'kld'.")

        loss /= input.shape[0]
        return loss


def kde(pdf: Tensor, sigma: float = DEFAULT_KDE_SIGMA):
    phi = phi_kernel(pdf.shape[-1], sigma, dtype=pdf.dtype, device=pdf.device)
    return (pdf[:, :, None] * phi[None, :, :]).sum(dim=1)


@cache
def phi_kernel(width: int, sigma: float, dtype: torch.dtype, device: torch.device):
    x = torch.arange(width, dtype=dtype, device=device)
    phi = torch.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    phi /= sigma * math.sqrt(2 * math.pi)
    return phi


class ScalarNet(nn.Module):
    def __init__(
        self,
        backbone: TBackbone,
        max_seq_len: int,
        norm: bool = False,
        norm_module: nn.Module = RMSNorm,
        activation: bool = True,
        activation_module: nn.Module = nn.SiLU,
    ):
        super().__init__()

        layers = [
            norm_module(backbone.input_dim) if norm else None,
            backbone,
            activation_module() if activation else None,
            nn.Linear(backbone.ouput_dim, 1),
        ]

        self.max_seq_len = max_seq_len
        self.layers = nn.Sequential(*filter(lambda l: l is not None, layers))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = F.sigmoid(x) * (self.max_seq_len + 1)
        return x

    @staticmethod
    def loss_fn(input: Tensor, target: Tensor, max_seq_len: int = None):

        loss = F.mse_loss(input, target, reduction="sum")
        loss /= input.shape[0]
        if max_seq_len is not None:
            loss /= (max_seq_len + 1) ** 2
        return loss


class BertBasedDistributionNet(nn.Module):
    def __init__(self, max_seq_len: int, pretrained_model_name: str = "bert-base-uncased"):
        """
        A BERT-based model with an MLP head for sequence distribution tasks.
        Reference: https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/bert#transformers.BertModel.forward.example
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(self.bert.config.hidden_size, max_seq_len + 1)
        )

    def forward(self, inputs: list[str]) -> Tensor:
        x = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(
            self.bert.device
        )

        outputs = self.bert(**x)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]  # CLS token embedding

        return self.mlp(pooled_output)

    @staticmethod
    def loss_fn(
        input: Tensor,
        target: Tensor,
        kde_sigma: float = DEFAULT_KDE_SIGMA,
        loss_type: str = DEFAULT_LOSS_TYPE,
    ):
        input_pdf = F.softmax(input, dim=1)
        target_pdf = kde(target, sigma=kde_sigma)

        if loss_type == "mse":
            loss = F.mse_loss(input_pdf, target_pdf, reduction="sum")
        elif loss_type == "kld":
            loss = F.kl_div(input_pdf.log(), target_pdf, reduction="sum")
        else:
            raise ValueError("Unsupported loss type. Choose 'mse' or 'kld'.")

        loss /= input.shape[0]
        return loss

    def get_optimizer(self, learning_rate: float = 1e-5, weight_decay: float = 0.01):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        return optimizer
