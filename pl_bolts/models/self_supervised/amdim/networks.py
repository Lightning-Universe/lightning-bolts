import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AMDIMEncoder(nn.Module):

    def __init__(
        self,
        dummy_batch,
        num_channels=3,
        encoder_feature_dim=64,
        embedding_fx_dim=512,
        conv_block_depth=3,
        encoder_size=32,
        use_bn=False
    ):
        super().__init__()
        # NDF = encoder hidden feat size
        # RKHS = output dim
        n_depth = conv_block_depth
        ndf = encoder_feature_dim
        self.ndf = encoder_feature_dim
        n_rkhs = embedding_fx_dim
        self.n_rkhs = embedding_fx_dim
        self.use_bn = use_bn
        self.dim2layer = None
        self.encoder_size = encoder_size

        # encoding block for local features
        print(f'Using a {encoder_size}x{encoder_size} encoder')
        if encoder_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError(f"Could not build encoder. Encoder size {encoder_size} is not supported")
        self._config_modules(dummy_batch, output_widths=[1, 5, 7], n_rkhs=n_rkhs, use_bn=use_bn)

    def init_weights(self, init_scale=1.):
        """
        Run custom weight init for modules...
        """
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, output_widths, n_rkhs, use_bn):
        """
        Configure the modules for extracting fake rkhs embeddings for infomax.
        """
        # get activations from each block to see output dims
        enc_acts = self._forward_acts(x)

        # out dimension to layer index
        # dim = number of output feature vectors
        self.dim2layer = {}

        # pull out layer indexes for the requested output_widths
        for layer_i, conv_out in enumerate(enc_acts):
            for output_width in output_widths:
                b, c, w, h = conv_out.size()
                if w == output_width:
                    self.dim2layer[w] = layer_i

        # get projected activation sizes at different layers
        # ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        ndf_7 = enc_acts[self.dim2layer[7]].size(1)

        # configure modules for fake rkhs embeddings
        self.rkhs_block_5 = FakeRKHSConvNet(ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        """
        Return activations from all layers.
        """
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)

        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        # compute activations in all layers for x
        activations = self._forward_acts(x)

        # gather rkhs embeddings from certain layers
        # last feature map with (b, d, 1, 1) (ie: last network out)
        r1 = activations[self.dim2layer[1]]

        # last feature map with (b, d, 5, 5)
        r5 = activations[self.dim2layer[5]]
        r5 = self.rkhs_block_5(r5)

        # last feature map with (b, d, 7, 7)
        r7 = activations[self.dim2layer[7]]
        r7 = self.rkhs_block_7(r7)

        return r1, r5, r7


class Conv3x3(nn.Module):

    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad, use_bn=True, pad_mode='constant'):
        super(Conv3x3, self).__init__()
        assert (pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0, bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn)

    def forward(self, x):
        if self.n_pad[0] > 0:
            # maybe pad the input
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # always apply conv
        x = self.conv(x)
        # maybe apply batchnorm
        x = self.bn(x)
        # always apply relu
        out = self.relu(x)
        return out


class ConvResBlock(nn.Module):

    def __init__(self, n_in, n_out, width, stride, pad, depth, use_bn):
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_bn)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_bn))
        self.layer_list = nn.Sequential(*layer_list)

    def init_weights(self, init_scale=1.):
        """
        Do a fixup-ish init for each ConvResNxN in this block.
        """
        for m in self.layer_list:
            m.init_weights(init_scale)

    def forward(self, x):
        # run forward pass through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out


class ConvResNxN(nn.Module):

    def __init__(self, n_in, n_out, width, stride, pad, use_bn=False):
        super(ConvResNxN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.n_grow = n_out - n_in
        if self.n_grow < 0:
            # use self.conv3 to downsample feature dim
            self.conv3 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=True)
        else:
            # self.conv3 is not used when n_out >= n_in
            self.conv3 = None
        self.bn1 = MaybeBatchNorm2d(n_out, True, use_bn)

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h2 = self.conv2(self.relu2(h1))
        if self.n_out < self.n_in:
            h3 = self.conv3(x)
        elif self.n_in == self.n_out:
            h3 = F.avg_pool2d(x, self.width, self.stride, self.pad)
        else:
            h3_pool = F.avg_pool2d(x, self.width, self.stride, self.pad)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.n_grow))
        h23 = h2 + h3
        return h23


class MaybeBatchNorm2d(nn.Module):

    def __init__(self, n_ftr, affine, use_bn):
        super(MaybeBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class NopNet(nn.Module):

    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x


class FakeRKHSConvNet(nn.Module):

    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = MaybeBatchNorm2d(n_output, True, use_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1, padding=0, bias=True)
        # when possible, initialize shortcut to be like identity
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.bool)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h
