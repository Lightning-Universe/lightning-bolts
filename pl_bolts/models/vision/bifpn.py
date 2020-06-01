import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.jit.annotations import Tuple, List, Dict

import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, ExtraFPNBlock

from timm.models.layers import create_conv2d, drop_path
from timm.models.layers.activations import Swish


class BiFPNBlock(nn.Module):
    """ BiFPNBlocl: Bi-directional Feature Pyramid Network block implementation
    with cross-scale connection and weighted feature fusion.

    Paper: EfficientDet - Scalable and Efficient Object Detection
    Reference: https://arxiv.org/abs/1911.09070, page 3

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted

    Examples::
        > m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        > # get some dummy data
        > x = OrderedDict()
        > x['feat0'] = torch.rand(1, 10, 64, 64) # e.g. P5
        > x['feat2'] = torch.rand(1, 20, 16, 16) # e.g. P4
        > x['feat3'] = torch.rand(1, 30, 8, 8)   # e.g. P3
        > # compute the FPN on top of x
        > output = m(x)
        > print([(k, v.shape) for k, v in output.items()])
        > # returns
        >   [('feat0', torch.Size([1, 5, 64, 64])),
        >    ('feat2', torch.Size([1, 5, 16, 16])),
        >    ('feat3', torch.Size([1, 5, 8, 8]))]
    """
    _invalid_input_size_msg = "BiFPN: Bad params, input `multi_scale_x` has {0} tensors but BiFPNBlock specified for {1} inputs"
        
    def __init__(self, in_channels_list, out_channels, conv_layer_class=None):
        super(BiFPN, self).__init__()
        
        # saving arguments in case for debugging
        self._in_channels_list = in_channels_list
        self._out_channels = out_channels

        self.topdown_blocks = nn.ModuleList()
        self.bottomup_blocks = nn.ModuleList()

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)

        def downsample_layer(x):
            h, w = x[-2:]
            extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
            extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

            left, right, top, bottom = extra_h // 2, extra_h - left, extra_v // 2, extra_v - top
            x = F.pad(x, [left, right, top, bottom])
            x = self.maxpool2d(x)
            return x

        self.downsample_layer = downsample_layer

        # Build Top Down and Bottom up connection connections. The connections are bidirectional consisting of
        # topdown and bottomup connections.
        # See reference: https://arxiv.org/abs/1911.09070, page 3, Figure 2(d)
        for in_channels in in_channels_list[1:-1]:
            if in_channels == 0:
                raise ValueError("BiFPN: Bad params, in_channels = 0 "
                    "in list: `in_channels_list` not supported.")
            # e.g. inner_blocks: P6_down(P7in, P6in), P5_down(P6_down, P5in), P4_down(P5_down, P4in)
            inner_block = self._create_depthwise_seperable_conv(in_channels, in_channels)
            self.topdown_blocks.append(inner_block)

        for in_channels in in_channels_list:
            # P4(P4_down, P3in, P4in), P5_down(P5_down, P4, P5in), P6(P6_down, P5, P6in), P7(P7_down, P6, P7in)
            inner_block = self._create_depthwise_seperable_conv(in_channels, out_channels)
            self.bottomup_blocks.append(inner_block)

        # weight initialisation using kaiming method
        self.initialize_params()
    
    def initialize_weight_params(self):
        pass

    def forward(self, multi_scale_x):
        """ Forward pass input from multi-scale features into bi-diretional feature pyramid block.
        Reference: https://arxiv.org/abs/1911.09070, page 3, Figure 2(d)
        """
        error_msg = _invalid_input_size_msg.format(len(multi_scale_x), len(self._in_channels_list))
        assert len(multi_scale_x) == len(self._in_channels_list), error_msg
        
        if isinstance(multi_scale_x, OrderedDict):
            multi_scale_x = list(x.values())

        topdown_outs = []  # to be populated by N-2 output, N = len(multi_scale_x)
        bottomup_outs = [] # to be populated by N output, N = len(multi_scale_x)
        
        topdown_last_out = multi_scale_x[-1]

        # Topdown Feedforward, the layer ii = 0 correspond to input to sideways P3 in the diagram
        # and topdown connection goes from P7 to P3 layer, hence layer ii = -1 to ii = 0
        # e.g. inner_blocks: P6_down(P7in, P6in), P5_down(P6_down, P5in), P4_down(P5_down, P4in)
        for ii in reversed(range(1, len(multi_scale_x)-1)):
            layer_input = multi_scale_x[ii] # the input features P_in_ii
            feat_shape = layer_input.shape[-2:]
            
            topdown_in = F.interpolate(topdown_last_out, scale_factor=2, mode='nearest')
            topdown_last_out = self.bottomup_blocks(topdown_in + layer_input)
            topdown_outs.append(topdown_last_out)
        
        bottomup_last_out = self.bottomup_blocks[0](multi_scale_x[0] + topdown_last_out)
        bottomup_outs.append(bottomup_last_out)

        # Bottomup Feedforward, in similar manner to the above.
        # Refer to the diagram in reference, Figure 2(d)
        for ii in range(1, len(multi_scale_x)):
            # Input for bottomup convolution layer is the sum of the following: 
            # * P_up_ii: previous bottomup connection output, 
            # * P_down_ii: output from topdown connection in same level
            # * P_in: skipped input in the same level
            bottomup_out = self.downsample_layer(bottomup_last_out)
            topdown_out = topdown_outs.pop()
            bottomup_last_out = self.bottomup_blocks[ii](multi_scale_x[ii] + topdown_out + bottomup_out)
            bottomup_outs.append(bottomup_last_out)

        # Example, feat0, torch.Tensor([...]) -> where tensor is output of layer 0
        # Here, layer 0 correspond to earlier layer e.g. P3 in the diagram
        # while layer N-1 correspond to deeper layer e.g. P7
        forward_outs = OrderedDict([
            ('feat{0}'.format(ii), feature)
            for ii, feature in enumerate(bottomup_outs)
        ])

        return forward_outs

    @staticmethod
    def _create_depthwise_seperable_conv(in_channels, out_channels, kernel_size=3, stride=2):
        depthwise_conv = create_conv2d(
            in_channels, in_channels, 3, stride=stride, groups=in_channels,
            padding='same', bias=False
        )
        pointwise_conv = create_conv2d(
            in_channels, out_channels, 1, stride=stride, groups=in_channels,
            padding='same'
        )
        return nn.Sequential(
            Swish(),
            depthwise_conv,
            pointwise_conv,
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        )
