# Based on https://github.com/pytorch/examples/blob/master/dcgan/main.py
from torch import nn


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, feature_maps, image_channels):
        super().__init__()
        self.gen = nn.Sequential(
            self._make_gen_block(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps, 4),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, last_block=False):
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )

        return gen_block

    def forward(self, noise):
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):
    def __init__(self, feature_maps, image_channels):
        super().__init__()
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True),
        )

    @staticmethod
    def _make_disc_block(
        in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, batch_norm=True, last_block=False
    ):
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, noise):
        return self.disc(noise)
