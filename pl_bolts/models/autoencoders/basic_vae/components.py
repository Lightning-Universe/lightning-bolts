import torch
from torch import nn
from torch.nn import functional as F


class Encoder(torch.nn.Module):
    """
    Takes as input an image, uses a CNN to extract features which
    get split into a mu and sigma vector
    """

    def __init__(self, hidden_dim, latent_dim, input_channels, input_width, input_height):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.c1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        conv_out_dim = self._calculate_output_dim(input_width, input_height)

        self.fc1 = DenseBlock(conv_out_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, hidden_dim)

        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.sigma_fc = nn.Linear(hidden_dim, latent_dim)

    def _calculate_output_dim(self, input_width, input_height):
        x = torch.rand(1, self.input_channels, input_width, input_height)
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

        # generate mu
        mu = self.mu_fc(x)

        # generate sigma
        sigma = self.sigma_fc(x)
        return mu, sigma


class Decoder(torch.nn.Module):
    """
    Takes in latent vars and reconstructs an image
    """

    def __init__(self, hidden_dim, latent_dim, input_width, input_height, input_channels):
        super().__init__()

        self.input_channels = input_channels
        self.deconv_dim_w, self.deconv_dim_h = self._calculate_output_size(input_width, input_height)

        self.latent_dim = latent_dim
        self.fc1 = DenseBlock(latent_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, self.deconv_dim_w * self.deconv_dim_h * 64)
        self.dc1 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.dc2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.dc3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, input_channels, kernel_size=1, stride=1)

    def _calculate_output_size(self, input_width, input_height):
        x = torch.rand(1, self.input_channels, input_width, input_height)
        dc1 = nn.Conv2d(self.input_channels, 32, kernel_size=1, stride=1)
        dc2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        dc3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        dc4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        x = dc4(dc3(dc2(dc1(x))))
        return x.size(-2), x.size(-1)

    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = x.view(x.size(0), 64, self.deconv_dim_w, self.deconv_dim_h)
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        x = F.relu(self.dc3(x))
        x = self.dc4(x)
        x = x.view(x.size(0), -1)
        return x


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
