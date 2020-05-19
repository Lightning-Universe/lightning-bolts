import torch
from torch import nn
from torch.nn import functional as F


class AEEncoder(torch.nn.Module):
    """
    Takes as input an image, uses a CNN to extract features which
    get split into a mu and sigma vector
    """

    def __init__(self, hidden_dim, latent_dim, input_width, input_height):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_width = input_width
        self.input_height = input_height

        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        conv_out_dim = self._calculate_output_dim(input_width, input_height)

        self.fc1 = DenseBlock(conv_out_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, hidden_dim)

        self.fc_z_out = nn.Linear(hidden_dim, latent_dim)

    def _calculate_output_dim(self, input_width, input_height):
        x = torch.rand(1, 1, input_width, input_height)
        x = self.c3(self.c2(self.c1(x)))
        x = x.view(-1)
        return x.size(0)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        z = self.fc_z_out(x)
        return z


class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_p=0.2):
        super().__init__()
        self.drop_p = drop_p
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc_bn = nn.BatchNorm1d(out_dim)
        self.in_dim = in_dim

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p)
        return x
