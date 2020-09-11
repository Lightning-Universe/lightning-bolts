from torchvision.datasets import CIFAR10, STL10

from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
from pl_bolts.datamodules.binary_mnist_datamodule import BinaryMNISTDataModule
from pl_bolts.datamodules.cifar10_datamodule import (
    CIFAR10DataModule,
    TinyCIFAR10DataModule,
)
from pl_bolts.datamodules.dummy_dataset import DummyDataset, DummyDetectionDataset
from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.sklearn_datamodule import (
    SklearnDataset,
    SklearnDataModule,
    TensorDataset,
    TensorDataModule,
)
from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.datamodules.vocdetection_datamodule import VOCDetectionDataModule
