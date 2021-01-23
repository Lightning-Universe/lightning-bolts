from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from torchdyn.models import NeuralDE, DataControl


class DepthInvariantNeuralODE(pl.LightningModule):
    """
    Vanilla Neural ODE with constant-in-depth parameters.
    Can be equipped with data control as described in https://arxiv.org/abs/2002.08071

    Example::
        TO DO

    Example CLI::
        TO DO
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            absolute_tol: float = 1e-4,
            relative_tol: float = 1e-4,
            integration_span: Tensor = torch.linspace(0, 1),
            data_control: str = 'True',
            lr: float = 0.01,
            **kwargs
    ):
        """
        Args:
            input_dim: size of input data
            hidden_dim: size of hidden state
            absolute_tol: absolute tolerance of the numerical solver
            relative_tol: relative tolerance of the numerical solver
            integration_span: mesh grid of evaluation points for the ODE
            data_control: data control mechanism as described in https://arxiv.org/abs/2002.08071
            learning_rate: learning rate for Adam

        """
        super().__init__()
        self.lr = lr

        if data_control:
            vector_field = nn.Sequential(DataControl(),
                                         nn.Linear(2 * input_dim, hidden_dim),
                                         nn.Softplus(),
                                         nn.Linear(hidden_dim, input_dim)
                                         )

        else:
            vector_field = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         nn.Softplus(),
                                         nn.Linear(hidden_dim, input_dim)
                                         )

        self.nde = NeuralDE(vector_field,
                             order=1,
                             sensitivity='adjoint',
                             solver='dopri5',
                             atol=absolute_tol,
                             rtol=relative_tol,
                             s_span=integration_span
                             )

    def forward(self, z):
        """
        Solve the NeuralODE in its integration interval with `z0` as initial condition,
        obtaining data features `zS` at final depth `S`

        Example::
            z0 = torch.rand(batch_size, input_dim)
            nde = DepthInvariantNeuralODE(batch_size, input_dim)
            zS = nde(z0)
        """
        return self.nde(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.nde(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        nfe = self.nde.nfe
        self.nde.nfe = 0
        self.log('loss', loss)
        return {'loss': loss, 'nfe': nfe}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [opt_g], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0002, help="adam: learning rate")
        return parser


def cli_main(args=None):
    from torchdyn.datasets import ToyDataset, generate_concentric_spheres, generate_moons
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="concentric_circles", type=str, help="concentric_circles, moons")
    script_args, _ = parser.parse_known_args(args)

    return None, None, None


if __name__ == '__main__':
    _, _, _ = cli_main()
