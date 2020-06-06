import torch
from torchvision import models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchvision_ssl_encoder(name, pretrained=False):
    pretrained_model = getattr(models, name)(pretrained=pretrained)

    pretrained_model.fc = Identity()
    return pretrained_model