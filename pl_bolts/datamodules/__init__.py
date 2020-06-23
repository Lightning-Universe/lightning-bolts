from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule, TinyCIFAR10DataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset, SklearnDataModule


DATAMODULE_COLLECTION = {
    'tiny-cifar10': TinyCIFAR10DataModule,
    'cifar10': CIFAR10DataModule,
    'stl10': STL10DataModule,
    'imagenet2012': SSLImagenetDataModule,
}


def get_datamodule(name, data_dir, num_workers, **kwargs):
    if name not in DATAMODULE_COLLECTION:
        raise FileNotFoundError(f'the {name} dataset is not supported.'
                                ' Subclass "get_dataset to provide your own"')
    return DATAMODULE_COLLECTION[name](data_dir=data_dir, num_workers=num_workers, **kwargs)