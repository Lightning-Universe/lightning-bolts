"""
PixelCNN
Implemented by: William Falcon
Reference: https://arxiv.org/pdf/1905.09272.pdf (page 15)
Accessed: May 14, 2020
"""
from torch import nn
from torch.nn import functional as F


class PixelCNN(nn.Module):
    """Implementation of `Pixel CNN <https://arxiv.org/abs/1606.05328>`_.

    Paper authors: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves,
    Koray Kavukcuoglu

    Implemented by:

        - William Falcon

    Example::

        >>> from pl_bolts.models.vision import PixelCNN
        >>> import torch
        ...
        >>> model = PixelCNN(input_channels=3)
        >>> x = torch.rand(5, 3, 64, 64)
        >>> out = model(x)
        ...
        >>> out.shape
        torch.Size([5, 3, 64, 64])
    """

    def __init__(self, input_channels: int, hidden_channels: int = 256, num_blocks=5):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.blocks = nn.ModuleList([self.conv_block(input_channels) for _ in range(num_blocks)])

    def conv_block(self, input_channels):
        c1 = nn.Conv2d(in_channels=input_channels, out_channels=self.hidden_channels, kernel_size=(1, 1))
        act1 = nn.ReLU()
        c2 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(1, 3))
        pad = nn.ConstantPad2d((0, 0, 1, 0, 0, 0, 0, 0), 1)
        c3 = nn.Conv2d(
            in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(2, 1), padding=(0, 1)
        )
        act2 = nn.ReLU()
        c4 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=input_channels, kernel_size=(1, 1))

        block = nn.Sequential(c1, act1, c2, pad, c3, act2, c4)
        return block

    def forward(self, z):
        c = z
        for conv_block in self.blocks:
            c = c + conv_block(c)

        c = F.relu(c)
        return c
