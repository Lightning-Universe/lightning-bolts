import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

'''
    As in the paper, the wide resnet only considers the resnet of the pre-activated version, 
    and it only considers the basic blocks rather than the bottleneck blocks.
'''


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=0.001)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_chan, momentum=0.001)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.pre_res_act = pre_res_act

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu1(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = act1 if self.pre_res_act else x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out = shortcut + residual
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class WideResnetBackbone(nn.Module):
    def __init__(self, k=1, n=28, drop_rate=0):
        super(WideResnetBackbone, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(3)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1,
            drop_rate=drop_rate,
            pre_res_act=True,
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.init_weight()

    def create_layer(self, input_channel, out_channel, bnum,
                     stride=1, drop_rate=0,
                     pre_res_act=False):
        layers = [
            BasicBlockPreAct(
                input_channel,
                out_channel,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_channel,
                    out_channel,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat)  # 1/2
        feat4 = self.layer3(feat2)  # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)
        return feat2, feat4

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class WideResnet(nn.Module):
    '''
    for wide-resnet-28-10, the definition should be WideResnet(n_classes, 10, 28)
    '''

    def __init__(self, n_classes, k=1, n=28, low_dim=64, proj=False):
        super(WideResnet, self).__init__()
        self.n_layers, self.k = n, k
        self.backbone = WideResnetBackbone(k=k, n=n)
        self.classifier = nn.Linear(64 * self.k, n_classes, bias=True)

        self.proj = proj
        if proj:
            self.l2norm = Normalize(2)
            self.fc1 = nn.Linear(64 * self.k, 64 * self.k)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(64 * self.k, low_dim)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        out = self.classifier(feat)

        if self.proj:
            feat = self.fc1(feat)
            feat = self.relu_mlp(feat)
            feat = self.fc2(feat)

            feat = self.l2norm(feat)
            return out, feat
        else:
            return out

    def init_weight(self):
        nn.init.xavier_normal_(self.classifier.weight)
        if not self.classifier.bias is None:
            nn.init.constant_(self.classifier.bias, 0)


def get_ema_model(model):
    ema_model = deepcopy(model)
    for q, k in zip(model.parameters(), ema_model.parameters()):
        k.data.copy_(q.detach().data)
        # DO NOT UPDATE for eval.
        k.requires_grad = False
    ema_model.eval()
    return ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)


def wide_resnet_28_10(n_classes):
    return WideResnet(n_classes, 10, 28)
