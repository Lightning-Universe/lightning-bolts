import torch
from torch import nn


def center_crop(image, new_shape):
    h, w = image.shape[-2:]
    n_h, n_w = new_shape[-2:]
    cy, cx = int(h / 2), int(w / 2)
    xmin, ymin = cx - n_w // 2, cy - n_h // 2
    xmax, ymax = xmin + n_w, ymin + n_h
    cropped_image = image[..., xmin:xmax, ymin:ymax]
    return cropped_image


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class UpSampleConv(nn.Module):

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):

        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = center_crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, use_dropout=False, use_bn=True):
        super(self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(in_channels * 2)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

        self.conv_block1 = ConvBlock(in_channels, in_channels * 2, use_dropout, use_bn)
        self.conv_block2 = ConvBlock(in_channels * 2, in_channels * 2, use_dropout, use_bn)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.maxpool(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, depth=6):
        super(self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.conv_final = nn.Conv2d(hidden_channels,
                                    out_channels,
                                    kernel_size=1)
        self.depth = depth

        self.contracting_layers = []
        self.expanding_layers = []
        self.sigmoid = nn.Sigmoid()

        # encoding/contracting path of the Generator
        for i in range(depth):
            self.contracting_layers += [
                DownSampleConv(hidden_channels * 2 ** i, use_dropout=True if i < 3 else False)
            ]

        # Upsampling/Expanding path of the Generator
        for i in range(depth):
            self.expanding_layers += [UpSampleConv(hidden_channels * 2 ** (i + 1))]

        self.contracting_layers = nn.ModuleList(self.contracting_layers)
        self.expanding_layers = nn.ModuleList(self.expanding_layers)

    def forward(self, x):
        depth = self.depth
        contractive_x = []

        x = self.conv1(x)
        contractive_x.append(x)

        for i in range(depth):
            x = self.contracting_layers[i](x)
            print(x.shape)
            contractive_x.append(x)

        for i in range(depth - 1, -1, -1):
            x = self.expanding_layers[i](x, contractive_x[i])
            print(x.shape)
        x = self.conv_final(x)

        return self.sigmoid(x)


class Discriminator(nn.Module):

    def __init__(self, input_channels, hidden_channels=8):
        `super().__init__()`
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.contract1 = DownSampleConv(hidden_channels, use_bn=False)
        self.contract2 = DownSampleConv(hidden_channels * 2)
        self.contract3 = DownSampleConv(hidden_channels * 4)
        self.contract4 = DownSampleConv(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.conv1(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn
