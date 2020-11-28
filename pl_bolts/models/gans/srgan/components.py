import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Residual block: k3n64s1 x2
        self.block = nn.Sequential(
            self._make_conv_block(channels),
            self._make_conv_block(channels, prelu=False),
        )

    @staticmethod
    def _make_conv_block(channels, prelu=True):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU() if prelu else nn.Identity(),
        )

    def forward(self, x):
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    def __init__(self, in_channels, feature_maps=64, num_res_blocks=16, num_ps_blocks=2):
        super().__init__()
        # k9n64s1 --> B x (k3n64s1 x 2) --> k3n64s1 --> 2 x k3n256s1 --> k9n3s1

        # Input block: k9n64s1
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # B residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [ResidualBlock(feature_maps)]

        # k3n64s1
        res_blocks += [
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_maps),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks
        ps_blocks = []
        for _ in range(num_ps_blocks):
            ps_blocks += [
                nn.Conv2d(feature_maps, 4 * feature_maps, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output block: k9n3s1
        self.output_block = nn.Sequential(
            nn.Conv2d(feature_maps, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        x_res = self.input_block(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.output_block(x)
        return x


class SRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, feature_maps=64, n_blocks=3):
        super().__init__()
        # k3n64s1 --> k3n64s2 --> k3n128s1 --> k3n128s2 --> k3n256s1 --> k3n256s2 --> k3n512s1 --> k3n512s2 --> MLP
        self.conv_blocks = nn.Sequential(
            # k3n64s1 --> k3n64s2
            self._make_double_conv_block(in_channels, feature_maps, first_batch_norm=False),
            # k3n128s1 --> k3n128s2
            self._make_double_conv_block(feature_maps, feature_maps * 2),
            # k3n256s1 --> k3n256s2
            self._make_double_conv_block(feature_maps * 2, feature_maps * 4),
            # k3n512s1 --> k3n512s2
            self._make_double_conv_block(feature_maps * 4, feature_maps * 8),
        )

        self.mlp += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, 1, kernel_size=1, padding=0),
            nn.Flatten(),
        ]

    @staticmethod
    def _make_conv_block(in_channels, out_channels, stride=1, batch_norm=True):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _make_double_conv_block(self, in_channels, out_channels, first_batch_norm=True):
        return nn.Sequential(
            self._make_conv_block(in_channels, out_channels, batch_norm=first_batch_norm),
            self._make_conv_block(out_channels, out_channels, stride=2),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.mlp(x)
        return x
