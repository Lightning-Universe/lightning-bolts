import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape, hidden_dim=256):
        super().__init__()
        feats = int(np.prod(img_shape))
        self.img_shape = img_shape
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, feats)

    # forward method
    def forward(self, z):
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        img = torch.tanh(self.fc4(z))
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self, img_shape, hidden_dim=1024):
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
