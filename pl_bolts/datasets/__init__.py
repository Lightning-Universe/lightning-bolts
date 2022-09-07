import urllib

from pl_bolts.datasets.array_dataset import ArrayDataset
from pl_bolts.datasets.base_dataset import DataModel, LightDataset
from pl_bolts.datasets.cifar10_dataset import CIFAR10, TrialCIFAR10
from pl_bolts.datasets.concat_dataset import ConcatDataset
from pl_bolts.datasets.dummy_dataset import (
    DummyDataset,
    DummyDetectionDataset,
    RandomDataset,
    RandomDictDataset,
    RandomDictStringDataset,
)
from pl_bolts.datasets.emnist_dataset import BinaryEMNIST
from pl_bolts.datasets.imagenet_dataset import UnlabeledImagenet, extract_archive, parse_devkit_archive
from pl_bolts.datasets.kitti_dataset import KittiDataset
from pl_bolts.datasets.mnist_dataset import MNIST, BinaryMNIST
from pl_bolts.datasets.ssl_amdim_datasets import CIFAR10Mixed, SSLDatasetMixin

__all__ = [
    "ArrayDataset",
    "DataModel",
    "LightDataset",
    "CIFAR10",
    "TrialCIFAR10",
    "ConcatDataset",
    "DummyDataset",
    "DummyDetectionDataset",
    "MNIST",
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
    "BinaryEMNIST",
]

# TorchVision hotfix https://github.com/pytorch/vision/issues/1938
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)
