import pytorch_lightning as pl
import torch
from torch import nn

from pl_bolts.models.gans.pix2pix.components import Generator, Discriminator


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Pix2Pix(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int = 32,
                 depth: int = 6,
                 learning_rate: float = 0.0002,
                 lambda_recon: int = 200):
        self.gen = Generator(in_channels, out_channels, hidden_channels, depth)
        self.disc = Discriminator(in_channels, hidden_channels=8)
        self.learning_rate = learning_rate

        # intializing weights
        self.gen = self.gen.apply(weights_init)
        self.disc = self.disc.apply(weights_init)

        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        self.lambda_recon = lambda_recon
