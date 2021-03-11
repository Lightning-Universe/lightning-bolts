import urllib

from pl_bolts.datasets.base_dataset import LightDataset
from pl_bolts.datasets.cifar10_dataset import CIFAR10, TrialCIFAR10
from pl_bolts.datasets.concat_dataset import ConcatDataset
from pl_bolts.datasets.dummy_dataset import (
    DummyDataset,
    DummyDetectionDataset,
    RandomDataset,
    RandomDictDataset,
    RandomDictStringDataset,
)
from pl_bolts.datasets.imagenet_dataset import extract_archive, parse_devkit_archive, UnlabeledImagenet
from pl_bolts.datasets.kitti_dataset import KittiDataset
from pl_bolts.datasets.mnist_dataset import BinaryMNIST
from pl_bolts.datasets.ssl_amdim_datasets import CIFAR10Mixed, SSLDatasetMixin

__all__ = [
    "LightDataset",
    "CIFAR10",
    "TrialCIFAR10",
    "ConcatDataset",
    "DummyDataset",
    "DummyDetectionDataset",
    "RandomDataset",
    "RandomDictDataset",
    "RandomDictStringDataset",
    "extract_archive",
    "parse_devkit_archive",
    "UnlabeledImagenet",
    "KittiDataset",
    "BinaryMNIST",
    "CIFAR10Mixed",
    "SSLDatasetMixin",
]

# TorchVision hotfix https://github.com/pytorch/vision/issues/1938
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
