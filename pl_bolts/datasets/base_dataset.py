import logging
import os
import urllib.request
from abc import ABC
from typing import Callable, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError

import numpy as np
import torch
from pytorch_lightning.utilities import exceptions
from torch import Tensor
from torch.utils.data import Dataset

from pl_bolts.utils.stability import under_review

# TODO: Move this to a more appropriate place.
ARRAYS = Union[torch.Tensor, np.ndarray, List[Union[float, int]]]


@under_review()
class LightDataset(ABC, Dataset):
    data: Tensor
    targets: Tensor
    normalize: tuple
    dir_path: str
    cache_folder_name: str
    DATASET_NAME = "light"

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.dir_path, self.DATASET_NAME, self.cache_folder_name)

    @staticmethod
    def _prepare_subset(
        full_data: Tensor,
        full_targets: Tensor,
        num_samples: int,
        labels: Sequence,
    ) -> Tuple[Tensor, Tensor]:
        """Prepare a subset of a common dataset."""
        classes = {d: 0 for d in labels}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float("inf")) >= num_samples:
                continue
            indexes.append(idx)
            classes[label] += 1
            if all(classes[k] >= num_samples for k in classes):
                break
        data = full_data[indexes]
        targets = full_targets[indexes]
        return data, targets

    def _download_from_url(self, base_url: str, data_folder: str, file_name: str):
        url = os.path.join(base_url, file_name)
        logging.info(f"Downloading {url}")
        fpath = os.path.join(data_folder, file_name)
        try:
            urllib.request.urlretrieve(url, fpath)
        except HTTPError as err:
            raise RuntimeError(f"Failed download from {url}") from err


class BoltsDataset(Dataset):
    """Base class for Bolts datasets.

    Args:
        arrays: Sequence of indexables.
        transforms: A callable that takes input data and target data and returns transformed versions of both.
        input_transform: A callable that takes input data and returns a transformed version.
        target_transform: A callable that takes target data and returns a transformed version.

    Raises:
        MisconfigurationException: Only transforms or input_transform/target_transform can be passed as argument.
        MisconfigurationException: input_transform and target_transform can only be applied to arrays length 2.
        MisconfigurationException: target_transform can only be applied to arrays length 2.
    """

    def __init__(
        self,
        *arrays: Tuple[ARRAYS],
        transforms: Optional[Callable] = None,
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        has_transforms = transforms is not None
        has_separate_transform = input_transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise exceptions.MisconfigurationException(
                "Only transforms or input_transform/target_transform can be passed as argument"
            )

        if input_transform and target_transform and len(arrays) != 2:
            raise exceptions.MisconfigurationException(
                "input_transform and target_transform can only be applied to arrays of length 2."
            )

        if target_transform and len(arrays) != 2:
            raise exceptions.MisconfigurationException("target_transform can only be applied to arrays of length 2.")

        self.arrays = arrays
        self.transforms = transforms
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
