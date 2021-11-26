"""Adapted from: https://github.com/https-deeplearning-ai/GANs-Public."""
import torch
import torch.nn as nn

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.models import vgg19
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class ResidualBlock(nn.Module):
    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.PReLU(),
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    def __init__(
        self,
        image_channels: int,
        feature_maps: int = 64,
        num_res_blocks: int = 16,
        num_ps_blocks: int = 2,
    ) -> None:
        super().__init__()
        # Input block (k9n64s1)
        self.input_block = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # B residual blocks (k3n64s1)
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [ResidualBlock(feature_maps)]

        # k3n64s1
        res_blocks += [
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks (k3n256s1)
        ps_blocks = []
        for _ in range(num_ps_blocks):
            ps_blocks += [
                nn.Conv2d(feature_maps, 4 * feature_maps, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output block (k9n3s1)
        self.output_block = nn.Sequential(
            nn.Conv2d(feature_maps, image_channels, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.input_block(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.output_block(x)
        return x


class SRGANDiscriminator(nn.Module):
    def __init__(self, image_channels: int, feature_maps: int = 64) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # k3n64s1, k3n64s2
            self._make_double_conv_block(image_channels, feature_maps, first_batch_norm=False),
            # k3n128s1, k3n128s2
            self._make_double_conv_block(feature_maps, feature_maps * 2),
            # k3n256s1, k3n256s2
            self._make_double_conv_block(feature_maps * 2, feature_maps * 4),
            # k3n512s1, k3n512s2
            self._make_double_conv_block(feature_maps * 4, feature_maps * 8),
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, 1, kernel_size=1),
            nn.Flatten(),
        )

    def _make_double_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        first_batch_norm: bool = True,
    ) -> nn.Sequential:
        return nn.Sequential(
            self._make_conv_block(in_channels, out_channels, batch_norm=first_batch_norm),
            self._make_conv_block(out_channels, out_channels, stride=2),
        )

    @staticmethod
    def _make_conv_block(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        batch_norm: bool = True,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.mlp(x)
        return x


class VGG19FeatureExtractor(nn.Module):
    def __init__(self, image_channels: int = 3) -> None:
        super().__init__()

        assert image_channels in [1, 3]
        self.image_channels = image_channels

        vgg = vgg19(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.image_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.vgg(x)
