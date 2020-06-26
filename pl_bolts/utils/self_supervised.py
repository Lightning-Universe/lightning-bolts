import torch
from pl_bolts.models.self_supervised import resnets


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchvision_ssl_encoder(name, pretrained=False, return_all_feature_maps=False):
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)

    pretrained_model.fc = Identity()
    return pretrained_model
