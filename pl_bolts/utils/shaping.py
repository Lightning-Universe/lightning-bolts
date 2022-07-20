import numpy as np
import torch
from torch import Tensor

from pl_bolts.utils.stability import under_review


@under_review()
def tile(a: Tensor, dim: int, n_tile: int) -> Tensor:
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
