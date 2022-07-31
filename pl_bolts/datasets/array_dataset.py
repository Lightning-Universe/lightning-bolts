from typing import Tuple

import torch
from pytorch_lightning.utilities import exceptions
from torch import Tensor
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """Dataset wrapping tensors, lists, numpy arrays.

    Args:
        data: feature variables.
        target: target variables.

    Raises:
        MisconfigurationException: if there is a shape mismatch between arrays in the first dimension.

    Example:
        >>> from sklearn.datasets import load_diabetes
        >>> from pl_bolts.datasets import ArrayDataset
        ...
        >>> X, y = load_diabetes(return_X_y=True)
        >>> dataset = ArrayDataset(X, y)
        >>> len(dataset)
        442
    """

    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target

        if not torch.is_tensor(self.data):
            self.data = torch.tensor(data)

        if not torch.is_tensor(target):
            self.target = torch.tensor(target)

        if not self.data.size(0) == self.target.size(0):
            raise exceptions.MisconfigurationException("Shape mismatch between arrays in the first dimension")

    def __len__(self) -> int:
        return self.target.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.data[idx].float()
        y = self.target[idx]

        if not ((y.dtype == torch.int32) or (y.dtype == torch.int64)):
            y = y.float()
        return x, y
