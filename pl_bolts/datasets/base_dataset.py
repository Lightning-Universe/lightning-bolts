import logging
import os
import urllib.parse
import urllib.request
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError

from torch import Tensor
from torch.utils.data import Dataset

from pl_bolts.utils.stability import under_review
from pl_bolts.utils.types import TArrays


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
        url = urllib.parse.urljoin(base_url, file_name)
        logging.info(f"Downloading {url}")
        fpath = os.path.join(data_folder, file_name)
        try:
            urllib.request.urlretrieve(url, fpath)
        except HTTPError as err:
            raise RuntimeError(f"Failed download from {url}") from err


@dataclass
class DataModel:
    """Data model dataclass.

    Ties together data and callable transforms.

    Attributes:
        data: Sequence of indexables.
        transform: Callable to transform data. The transform is called on a subset of data.
    """

    data: TArrays
    transform: Optional[Callable[[TArrays], TArrays]] = None

    def process(self, subset: Union[TArrays, float]) -> Union[TArrays, float]:
        """Transforms a subset of data.

        Args:
            subset: Sequence of indexables.

        Returns:
            data: Transformed data if transform is not None.
        """
        if self.transform is not None:
            subset = self.transform(subset)
        return subset
