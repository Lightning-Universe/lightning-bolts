from typing import Tuple, Union

from pytorch_lightning.utilities import exceptions
from torch.utils.data import Dataset

from pl_bolts.datasets.base_dataset import DataModel, TArrays


class ArrayDataset(Dataset):
    """Dataset wrapping tensors, lists, numpy arrays.

    Any number of ARRAYS can be inputted into the dataset. The ARRAYS are transformed on each `__getitem__`. When
        transforming, please refrain from chaning the hape of ARRAYS in the first demension.

    Attributes:
        data_models: Sequence of data models.

    Raises:
        MisconfigurationException: if there is a shape mismatch between arrays in the first dimension.

    Example:
        >>> from pl_bolts.datasets import ArrayDataset, DataModel
        >>> from pl_bolts.datasets.utils import to_tensor

        >>> features = DataModel(data=[[1, 0, -1, 2], [1, 0, -2, -1], [2, 5, 0, 3]], transform=to_tensor)
        >>> target = DataModel(data=[1, 0, 0], transform=to_tensor)

        >>> ds = ArrayDataset(features, target)
        >>> len(ds)
        3
    """

    def __init__(self, *data_models: DataModel) -> None:
        """Initialises class and checks if arrays are the same shape in the first dimension."""
        self.data_models = data_models

        if not self._equal_size():
            raise exceptions.MisconfigurationException("Shape mismatch between arrays in the first dimension")

    def __len__(self) -> int:
        return len(self.data_models[0].data)

    def __getitem__(self, idx: int) -> Tuple[Union[TArrays, float], ...]:
        return tuple(data_model.process(data_model.data[idx]) for data_model in self.data_models)

    def _equal_size(self) -> bool:
        """Checks the size of the data_models are equal in the first dimension.

        Returns:
            bool: True if size of data_models are equal in the first dimension. False, if not.
        """
        return len({len(data_model.data) for data_model in self.data_models}) == 1
