from typing import Optional, Tuple

import torch
from torch import nn

from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class MLP(nn.Module):

    def __init__(self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        input_dim: int = 2048,
        hidden_size: int = 4096,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
