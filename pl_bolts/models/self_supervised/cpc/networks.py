from torch import nn
from torch.nn import functional as F

from pl_bolts.utils.stability import under_review


@under_review()
class CPCResNet(nn.Module):
    def __init__(
        self,
        sample_batch,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer
        self.trainer = None
        self.experiment = None
        self.batch_size = None

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "`replace_stride_with_dilation` should be None"
                f" or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        # transform batch for LN
        sample_batch = self.conv1(sample_batch)
        self.ln1 = norm_layer(sample_batch.size()[1:])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        sample_batch = self.maxpool(sample_batch)

        self.layer1, sample_batch = self._make_layer(sample_batch, block, 64, blocks=layers[0])
        self.layer2, sample_batch = self._make_layer(
            sample_batch, block, 128, blocks=layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3, sample_batch = self._make_layer(
            sample_batch, block, 512, blocks=layers[2], stride=2, dilate=replace_stride_with_dilation[1], expansion=8
        )
        self.layer4, sample_batch = self._make_layer(
            sample_batch, block, 512, blocks=layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, sample_batch, block, planes, blocks, stride=1, dilate=False, expansion=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            conv = conv1x1(self.inplanes, planes * expansion, stride)
            downsample = conv

        layers = []
        layer = block(
            sample_batch,
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer,
            expansion,
        )

        sample_batch = layer(sample_batch)
        layers.append(layer)
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layer = block(
                sample_batch,
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                expansion=expansion,
            )
            sample_batch = layer(sample_batch)
            layers.append(layer)

        return nn.Sequential(*layers), sample_batch

    def flatten(self, x):
        x = x.view(self.batch_size, -1)
        x = F.avg_pool1d(x.unsqueeze(1), 4).squeeze(1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


@under_review()
def cpc_resnet101(sample_batch, **kwargs):
    return CPCResNet(sample_batch, LNBottleneck, [3, 4, 46, 3], **kwargs)


@under_review()
def cpc_resnet50(sample_batch, **kwargs):
    return CPCResNet(sample_batch, LNBottleneck, [3, 4, 6, 3], **kwargs)


@under_review()
class LNBottleneck(nn.Module):
    def __init__(
        self,
        sample_batch,
        inplanes,
        planes,
        stride=1,
        downsample_conv=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        expansion=4,
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        self.__init_layer_norms(sample_batch, self.conv1, self.conv2, self.conv3, downsample_conv)

    def __init_layer_norms(self, x, conv1, conv2, conv3, downsample_conv):
        if downsample_conv is not None:
            xa = downsample_conv(x)
            b, c, w, h = xa.size()
            ln4 = nn.LayerNorm([c, w, h])
            self.downsample = nn.Sequential(downsample_conv, ln4)

        x = conv1(x)
        self.ln1 = nn.LayerNorm(x.size()[1:])

        x = conv2(x)
        self.ln2 = nn.LayerNorm(x.size()[1:])

        x = conv3(x)
        self.ln3 = nn.LayerNorm(x.size()[1:])

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@under_review()
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


@under_review()
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
