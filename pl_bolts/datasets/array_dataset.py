from typing import Tuple

from pytorch_lightning.utilities import exceptions
from torch.utils.data import Dataset

from pl_bolts.datasets.base_dataset import ARRAYS, DataModel


class ArrayDataset(Dataset):
    """Dataset wrapping tensors, lists, numpy arrays.

    Attributes:
        data_models: Sequence of data models.

    Raises:
        MisconfigurationException: if there is a shape mismatch between arrays in the first dimension.
    """

    def __init__(self, *data_models: DataModel) -> None:
        self.data_models = data_models

        if not self._equal_size():
            raise exceptions.MisconfigurationException("Shape mismatch between arrays in the first dimension")

    def __len__(self) -> int:
        return len(self.data_models[0].data)

    def __getitem__(self, idx: int) -> Tuple[ARRAYS, ...]:
        return tuple(data_model.process(data_model.data[idx]) for data_model in self.data_models)

    def _equal_size(self) -> bool:
        """Checks the size of the data_models are equal in the first dimension.

        Returns:
            bool: True if size of data_models are equal in the first dimension. False, if not.
        """
        return all(len(data_model.data) == len(self.data_models[0].data) for data_model in self.data_models)
