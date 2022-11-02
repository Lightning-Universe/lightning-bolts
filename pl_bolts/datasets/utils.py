import gzip
import os
import tarfile
import zipfile
from typing import List, Optional

import torch
from torch.utils.data.dataset import random_split

from pl_bolts.datasets.sr_celeba_dataset import SRCelebA
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST
from pl_bolts.datasets.sr_stl10_dataset import SRSTL10
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.types import TArrays


@under_review()
def prepare_sr_datasets(dataset: str, scale_factor: int, data_dir: str):
    """Creates train, val, and test datasets for training a Super Resolution GAN.

    Args:
        dataset: string indicating which dataset class to use (celeba, mnist, or stl10).
        scale_factor: scale factor between low- and high resolution images.
        data_dir: root dir of dataset.

    Returns:
        sr_datasets: tuple containing train, val, and test dataset.
    """
    assert dataset in ["celeba", "mnist", "stl10"]

    if dataset == "celeba":
        dataset_cls = SRCelebA
        dataset_train = dataset_cls(scale_factor, root=data_dir, split="train", download=True)
        dataset_val = dataset_cls(scale_factor, root=data_dir, split="valid", download=True)
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test", download=True)

    elif dataset == "mnist":
        dataset_cls = SRMNIST
        dataset_dev = dataset_cls(scale_factor, root=data_dir, train=True, download=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(scale_factor, root=data_dir, train=False, download=True)

    elif dataset == "stl10":
        dataset_cls = SRSTL10
        dataset_dev = dataset_cls(scale_factor, root=data_dir, split="train", download=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test", download=True)

    return (dataset_train, dataset_val, dataset_test)


def to_tensor(arrays: TArrays) -> torch.Tensor:
    """Takes a sequence of type `TArrays` and returns a tensor.

    This function serves as a use case for the ArrayDataset.

    Args:
        arrays: Sequence of type `TArrays`

    Returns:
        Tensor of the integers
    """
    return torch.tensor(arrays)


def is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def safe_extract_tarfile(
    tar: tarfile.TarFile,
    path: str = ".",
    members: Optional[List[tarfile.TarInfo]] = None,
    *,
    numeric_owner: bool = False,
) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise RuntimeError(f"Attempted Path Traversal in Tar File {tar.name} with member: {member.name}")

    tar.extractall(path, members, numeric_owner=numeric_owner)


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    extracted = False
    for fn in (_extract_tar, _extract_gzip, _extract_zip):
        try:
            fn(from_path, to_path)
            extracted = True
            break
        except (tarfile.TarError, zipfile.BadZipfile, OSError):
            continue

    if not extracted:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)


def _extract_tar(from_path: str, to_path: str) -> None:
    with tarfile.open(from_path, "r:*") as tar:
        safe_extract_tarfile(tar, path=to_path)


def _extract_gzip(from_path: str, to_path: str) -> None:
    to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
    with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
        out_f.write(zip_f.read())


def _extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(from_path, "r") as z:
        z.extractall(to_path)
