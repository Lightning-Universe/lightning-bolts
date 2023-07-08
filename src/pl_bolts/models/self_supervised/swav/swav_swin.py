import math
from functools import partial
from types import FunctionType
from typing import Any, Callable, List, Optional, Union , Tuple

import packaging.version as pv
import torch
import torch.fx
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

# Support for functions not found in older versions of torchvision
if pv.parse(torchvision.__version__) >= pv.parse("0.13"):
    from torchvision.ops.misc import MLP, Permute
    from torchvision.ops.stochastic_depth import StochasticDepth
    from torchvision.utils import _log_api_usage_once
else:
    
    """The functions below are copied from the torchvision implementation."""

    if not hasattr(torchvision.utils, "_log_api_usage_once"):
        def _log_api_usage_once(obj: Any) -> None:
            """Logs API usage(module and name) within an organization. In a large ecosystem,
                    it's often useful to track the PyTorch and TorchVision APIs usage. This API provides
                    the similar functionality to the logging module in the Python stdlib.It can be used
                    for debugging purpose to log which methods are used and by default it is inactive,
                    unless the user manually subscribes a logger via the `SetAPIUsageLogger method
<https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
                    Please note it is triggered only once for the same API call within a process.
                    It does not collect any data from open-source users since it is no-op by default.

                    For more information, please refer to
                    * PyTorch note:
                    https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
                    * Logging policy:
                    https://github.com/pytorch/vision/issues/5052;

                    Args:
                        obj: an object to extract info from.
            """
            module = obj.__module__
            if not module.startswith("torchvision"):
                module = f"torchvision.internal.{module}"
            name = obj.__class__.__name__
            if isinstance(obj, FunctionType):
                name = obj.__name__
            torch._C._log_api_usage_once(f"{module}.{name}")
    if not hasattr(torchvision.ops, "stochastic_depth"):
        def stochastic_depth(input: Tensor, prob: float, mode: str, training: bool = True) -> Tensor:
            """Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
            <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual branches
            of residual architectures.

            Args:
                input: The input tensor or arbitrary dimensions with the first one
                        being its batch i.e. a batch with ``N`` rows.
                prob: probability of the input to be zeroed.
                mode: ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes randomly
                    selected rows from the batch.
                training: apply stochastic depth if is ``True``. Default: ``True``

            Returns:
                Tensor[N, ...]: The randomly zeroed tensor.
            """
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                _log_api_usage_once(stochastic_depth)
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"drop probability has to be between 0 and 1, but got {prob}")
            if mode not in ["batch", "row"]:
                raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
            if not training or prob == 0.0:
                return input

            survival_rate = 1.0 - prob
            size = [input.shape[0]] + [1] * (input.ndim - 1) if mode == "row" else [1] * input.ndim
            noise = torch.empty(size, dtype=input.dtype, device=input.device)
            noise = noise.bernoulli_(survival_rate)
            if survival_rate > 0.0:
                noise.div_(survival_rate)
            return input * noise
        
        torch.fx.wrap("stochastic_depth")

        class StochasticDepth(nn.Module):
            """See :func:`stochastic_depth`."""

            def __init__(self, prob: float, mode: str) -> None:
                super().__init__()
                _log_api_usage_once(self)
                self.prob = prob
                self.mode = mode

            def forward(self, input: Tensor) -> Tensor:
                return stochastic_depth(input, self.prob, self.mode, self.training)

            def __repr__(self) -> str:
                return f"{self.__class__.__name__}(p={self.prob}, mode={self.mode})"
    if not hasattr(torchvision.ops.misc, "MLP"):
        class MLP(torch.nn.Sequential):
            """This block implements the multi-layer perceptron (MLP) module.

            Args:
                in_channels: Number of channels of the input
                hidden_channels: List of the hidden channel dimensions
                norm_layer: Norm layer that will be stacked on top of the linear layer.
                    If ``None`` this layer won't be used.
                activation_layer:
                    Activation function which will be stacked on top of the normalization layer (if not None),
                    otherwise on top of the linear layer.
                    If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
                inplace: Parameter for the activation layer, which can optionally do the operation in-place.
                    Default is ``None``, which uses the respective default values of the ``activation_layer``
                    and Dropout layer.
                bias: Whether to use bias in the linear layer. Default ``True``
                dropout: The probability for the dropout layer. Default: 0.0
            """

            def __init__(
                self,
                in_channels: int,
                hidden_channels: List[int],
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                inplace: Optional[bool] = None,
                bias: bool = True,
                dropout: float = 0.0,
            )-> None:
                # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
                # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
                params = {} if inplace is None else {"inplace": inplace}

                layers = []
                in_dim = in_channels
                for hidden_dim in hidden_channels[:-1]:
                    layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
                    if norm_layer is not None:
                        layers.append(norm_layer(hidden_dim))
                    layers.append(activation_layer(**params))
                    layers.append(torch.nn.Dropout(dropout, **params))
                    in_dim = hidden_dim

                layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
                layers.append(torch.nn.Dropout(dropout, **params))

                super().__init__(*layers)
                _log_api_usage_once(self)
    if not hasattr(torchvision.ops.misc, "Permute"):
        class Permute(torch.nn.Module):
            """This module returns a view of the tensor input with its dimensions permuted.

            Args:
                dims (List[int]): The desired ordering of dimensions
            """

            def __init__(self, dims: List[int]) -> None:
                super().__init__()
                self.dims = dims

            def forward(self, x: Tensor) -> Tensor:
                return torch.permute(x, self.dims)


# Support meshgrid indexing for older versions of torch
def meshgrid(*tensors: Union[Tensor, List[Tensor]], indexing: Optional[str] = None) -> Tuple:
    if pv.parse(torch.__version__) >= pv.parse("1.10.0"):
        return torch.meshgrid(*tensors, indexing=indexing)
    return torch.meshgrid(*tensors)


def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    h, w, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    return torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    n = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(n, n, -1)
    return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)


torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        return self.reduction(x)  # ... H/2 W/2 2*C


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x: input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return self.norm(x)


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
)->Tensor:
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    It supports both of shifted and non-shifted window.
    Args:
        input: The input tensor or 4-dimensions [N, H, W, C].
        qkv_weight: The weight tensor of query, key, value.
        proj_weight: The weight tensor of projection.
        relative_position_bias: The learned relative position bias added to attention.
        window_size: Window size.
        num_head: Number of attention heads.
        shift_size: Shift size for shifted window attention.
        attention_dropout: Dropout ratio of attention weight. Default: 0.0.
        dropout: Dropout ratio of output. Default: 0.0.
        qkv_bias: The bias tensor of query, key, value. Default: None.
        proj_bias: The bias tensor of projection. Default: None.
        logit_scale: Logit scale of cosine attention for Swin Transformer V2. Default: None.

    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    b, h, w, c = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_h, pad_w, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_h:
        shift_size[0] = 0
    if window_size[1] >= pad_w:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
    x = x.view(b, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_windows, window_size[0] * window_size[1], c)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (c // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_h, pad_w))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h_sli in h_slices:
            for w_sli in w_slices:
                attn_mask[h_sli[0] : h_sli[1], w_sli[0] : w_sli[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(b, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    # unpad features
    return x[:, :h, :w, :].contiguous()


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """See :func:`shifted_window_attention`."""

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """See :func:`shifted_window_attention_v2`."""

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self) -> None:
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1,2*Wh-1,2*Ww-1,2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        return 16 * torch.sigmoid(relative_position_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
        )


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for shifted window attention.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout: Dropout rate. Default: 0.0.
        attention_dropout: Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: Stochastic depth rate. Default: 0.0.
        norm_layer: Normalization layer.  Default: nn.LayerNorm.
        attn_layer: Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        return x + self.stochastic_depth(self.mlp(self.norm2(x)))


class SwinTransformerBlockV2(SwinTransformerBlock):
    """Swin Transformer V2 Block.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for shifted window attention.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout: Dropout rate. Default: 0.0.
        attention_dropout: Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: Stochastic depth rate. Default: 0.0.
        norm_layer: Normalization layer.  Default: nn.LayerNorm.
        attn_layer: Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        return x + self.stochastic_depth(self.norm2(self.mlp(x)))


class SwinTransformer(nn.Module):
    """Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using Shifted
    Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.

    Args:
        patch_size: Patch size.
        embed_dim: Patch embedding dimension.
        depths: Depth of each Swin Transformer layer.
        num_heads: Number of attention heads in different layers.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout: Dropout rate. Default: 0.0.
        attention_dropout: Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: Stochastic depth rate. Default: 0.1.
        num_classes: Number of classes for classification head. Default: 1000.
        block: SwinTransformer Block. Default: None.
        norm_layer: Normalization layer. Default: None.
        downsample_layer: Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        normalize=False,
        output_dim=0,
        hidden_mlp=0,
        num_prototypes=0,
        eval_mode=False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = output_dim

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None

        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_features, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_features, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_backbone(self, x) ->Tensor:
        x = self.padding(x)

        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        return self.flatten(x)

    def forward_head(self, x) -> Tensor:
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs) -> Tensor:
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, None
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])

            if next(self.parameters()).is_cuda:
                _out = self.forward_backbone(_out.cuda(non_blocking=True))
            else:
                _out = self.forward_backbone(_out)

            output = _out if start_idx == 0 else torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes) -> None:
        super().__init__()
        self.nmb_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x) -> Tensor:
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    **kwargs: Any,
) -> SwinTransformer:
    return SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )


def swin_s(**kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        **kwargs,
    )


def swin_b(**kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        **kwargs,
    )


def swin_v2_t(**kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


def swin_v2_s(**kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )


def swin_v2_b(**kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
