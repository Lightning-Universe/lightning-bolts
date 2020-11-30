from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules.stl10_sr_datamodule import STL10_SR_DataModule
from pl_bolts.models.gans.srgan.components import SRGANGenerator


class SRResNet(pl.LightningModule):
    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps: Number of feature maps to use
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.srresnet = SRGANGenerator(self.hparams.image_channels, self.hparams.feature_maps)

    def configure_optimizers(self):
        return torch.optim.Adam(self.srresnet.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        return self.srresnet(x)

    def training_step(self, batch, batch_idx):
        hr_image, lr_image = batch
        fake = self(lr_image)
        loss = F.mse_loss(hr_image, fake)
        self.log("loss", loss, on_epoch=True, prog_bar=True)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--image_channels", default=3, type=int)
        parser.add_argument("--feature_maps", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = STL10_SR_DataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SRResNet.add_model_specific_args(parser)
    args = parser.parse_args(args)

    model = SRResNet(**vars(args))
    dm = STL10_SR_DataModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SRImageLoggerCallback()])
    trainer.fit(model, dm)

    torch.save(model.srresnet, "srresnet.pt")

    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
