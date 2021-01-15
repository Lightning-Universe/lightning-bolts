import logging
import os
import urllib.request
from abc import ABC
from typing import Sequence, Tuple
from urllib.error import HTTPError

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LightDataset(ABC, Dataset):

    data: torch.Tensor
    targets: torch.Tensor
    normalize: tuple
    dir_path: str
    cache_folder_name: str
    DATASET_NAME = 'light'

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.dir_path, self.DATASET_NAME, self.cache_folder_name)

    @staticmethod
    def _prepare_subset(
        full_data: torch.Tensor,
        full_targets: torch.Tensor,
        num_samples: int,
        labels: Sequence,
    ) -> Tuple[Tensor, Tensor]:
        """Prepare a subset of a common dataset."""
        classes = {d: 0 for d in labels}
        indexes = []
        for idx, target in enumerate(full_targets):
            label = target.item()
            if classes.get(label, float('inf')) >= num_samples:
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
        logging.info(f'Downloading {url}')
        fpath = os.path.join(data_folder, file_name)
        try:
            urllib.request.urlretrieve(url, fpath)
        except HTTPError as err:
            raise RuntimeError(f'Failed download from {url}') from err
