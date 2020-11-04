from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl


class LambaLayer(nn.Module):

    def __init__(self, d, heads, len_k, len_v, len_u):
        """

        Args:
            d: input dim
            heads: num heads
            len_k: key/query depth
            len_v: value depth
            len_u: intra-depth
        """
        super().__init__()
        self.d = d
        self.heads = heads
        self.len_k = len_k
        self.len_v = len_v
        self.len_u = len_u

        # query
        self.Wq = nn.Conv2d(self.d, self.len_k * heads, 1, bias=False)
        self.bn_q = nn.BatchNorm2d(self.len_k * heads)

        # value
        self.Wv = nn.Conv2d(self.d, self.len_v * self.len_u, 1, bias=False)
        self.bn_v = nn.BatchNorm2d(self.len_v * self.len_u)

        # key
        self.Wk = nn.Conv2d(self.d, self.len_k * self.len_u, 1, bias=False)

    def forward(self, x, c):
        """
        In the paper:

        n = seq len
        m = context len
        d_in = input dim
        d_out = output dim

        (in self-attention, x=c)

        Args:
            x: inputs (batch, n, d_in)
            c: context (batch, m, d_in)

        Returns:
            y: output (batch, n, d_out)

        """
        # compute query: (batch, n, d) -> (batch, ...)
        q = self.Wq(x)
        q = self.bn_q(q)

        # compute value: (batch, m, d) -> (batch, ...)
        v = self.Wv(c)
        v = self.bn_v(v)

        # compute key: (batch, m, d) -> (batch, ...)
        k = self.Wk(c)
        k_tilde = F.softmax(k, dim=-1)
