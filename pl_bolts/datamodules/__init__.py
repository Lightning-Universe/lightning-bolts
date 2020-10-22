from pl_bolts.datamodules.async_dataloader import AsynchronousLoader

__all__ = []

try:
    from pl_bolts.datamodules.binary_mnist_datamodule import BinaryMNISTDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['BinaryMNISTDataModule']

try:
    from pl_bolts.datamodules.cifar10_datamodule import (
        CIFAR10DataModule,
        TinyCIFAR10DataModule,
    )
except ModuleNotFoundError:
    pass
else:
    __all__ += ['CIFAR10DataModule', 'TinyCIFAR10DataModule']

try:
    from pl_bolts.datamodules.experience_source import (
        ExperienceSourceDataset,
        ExperienceSource,
        DiscountedExperienceSource,
    )
except ModuleNotFoundError:
    pass
else:
    __all__ += ['ExperienceSourceDataset', 'ExperienceSource', 'DiscountedExperienceSource']

try:
    from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['FashionMNISTDataModule']

try:
    from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['ImagenetDataModule']

try:
    from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['MNISTDataModule']

try:
    from pl_bolts.datamodules.sklearn_datamodule import (
        SklearnDataset,
        SklearnDataModule,
        TensorDataset,
    )
except ModuleNotFoundError:
    pass
else:
    __all__ += ['SklearnDataset', 'SklearnDataModule', 'TensorDataset']

try:
    from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['SSLImagenetDataModule']

try:
    from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['STL10DataModule']

try:
    from pl_bolts.datamodules.vocdetection_datamodule import VOCDetectionDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['VOCDetectionDataModule']

try:
    from pl_bolts.datamodules.cityscapes_datamodule import CityscapesDataModule
except ModuleNotFoundError:  # pragma: no-cover
    pass
else:
    __all__ += ['CityscapesDataModule']

try:
    from pl_bolts.datasets.kitti_dataset import KittiDataset
except ModuleNotFoundError:
    pass
else:
    __all__ += ['KittiDataset']

try:
    from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
except ModuleNotFoundError:
    pass
else:
    __all__ += ['KittiDataModule']
