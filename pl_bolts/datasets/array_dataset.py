from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning.utilities import apply_func, exceptions

from pl_bolts.datasets.base_dataset import BoltsDataset

# TODO: Move this to a more appropriate place.
ARRAYS = Union[torch.Tensor, np.ndarray, List[Union[float, int]]]


class ArrayDataset(BoltsDataset):
    """Dataset wrapping tensors, lists, numpy arrays.

    Args:
        arrays: Sequence of indexables.
        transforms: A callable that takes input data and target data and returns transformed versions of both.
        input_transform: A callable that takes input data and returns a transformed version.
        target_transform: A callable that takes target data and returns a transformed version.

    Attributes:
        tensors: Input arrays transformed into tensors.

    Raises:
        MisconfigurationException: if there is a shape mismatch between arrays in the first dimension.
    """

    def __init__(
        self,
        *arrays: Tuple[ARRAYS],
        transforms: Optional[Callable] = None,
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            *arrays, transforms=transforms, input_transform=input_transform, target_transform=target_transform
        )
        self.tensors = apply_func.apply_to_collection(self.arrays, dtype=(np.ndarray, list), function=torch.tensor)

        if not self._equal_size():
            raise exceptions.MisconfigurationException("Shape mismatch between arrays in the first dimension")

    def __len__(self) -> int:
        return self.tensors[0].size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data, target = self.tensors[idx][0], self.tensors[idx][1]

        if self.transforms is not None:
            return self.transforms(data, target)
        elif self.input_transform is not None and self.target_transform is not None:
            return self.input_transform(data), self.target_transform(target)
        elif self.input_transform is not None:
            return self.input_transform(data)
        else:
            return tuple(tensor[idx] for tensor in self.tensors)

    def _equal_size(self) -> bool:
        """Check the size of the tensors are equal in the first dimension."""
        return all(tensor.size(0) == self.tensors[0].size(0) for tensor in self.tensors)
