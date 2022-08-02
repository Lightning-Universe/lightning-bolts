from typing import List, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.utilities import apply_func, exceptions
from torch import Tensor
from torch.utils.data import Dataset

ARRAYS = Union[torch.Tensor, np.ndarray, List[Union[float, int]]]


class ArrayDataset(Dataset):
    """Dataset wrapping tensors, lists, numpy arrays.

    Args:
        arrays: Arrays that have the same size in the first dimension.

    Attributes:
        tensors: Input arrays transformed into tensors.

    Raises:
        MisconfigurationException: if there is a shape mismatch between arrays in the first dimension.
    """

    def __init__(self, *arrays: Tuple[ARRAYS]) -> None:
        self.tensors = apply_func.apply_to_collection(arrays, dtype=(np.ndarray, list), function=torch.tensor)

        if not self._equal_size():
            raise exceptions.MisconfigurationException("Shape mismatch between arrays in the first dimension")

    def __len__(self) -> int:
        return self.tensors[0].size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        return tuple(tensor[idx] for tensor in self.tensors)

    def _equal_size(self):
        """Check the size of the tensors are equal in the first dimension."""
        return all(tensor.size(0) == self.tensors[0].size(0) for tensor in self.tensors)
