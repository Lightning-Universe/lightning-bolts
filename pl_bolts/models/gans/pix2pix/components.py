import torch
from torch import Tensor, nn


class UpSampleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchnorm: bool = True,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel=4, strides=2, padding=1)]

        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))

        if dropout:
            layers.append(nn.Dropout2d(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DownSampleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel=4, strides=2, padding=1)]

        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Paper details:

        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4x4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = nn.ModuleList(
            [
                DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
                DownSampleConv(64, 128),  # bs x 128 x 64 x 64
                DownSampleConv(128, 256),  # bs x 256 x 32 x 32
                DownSampleConv(256, 512),  # bs x 512 x 16 x 16
                DownSampleConv(512, 512),  # bs x 512 x 8 x 8
                DownSampleConv(512, 512),  # bs x 512 x 4 x 4
                DownSampleConv(512, 512),  # bs x 512 x 2 x 2
                DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
            ]
        )

        # decoder/upsample convs
        self.decoders = nn.ModuleList(
            [
                UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
                UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
                UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
                UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
                UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
                UpSampleConv(512, 128),  # bs x 128 x 64 x 64
                UpSampleConv(256, 64),  # bs x 64 x 128 x 128
            ]
        )
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return self.sigmoid(xn)
