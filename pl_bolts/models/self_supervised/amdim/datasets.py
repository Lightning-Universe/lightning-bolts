from torch.utils.data import random_split
from torchvision.datasets import STL10

from pl_bolts.datamodules.imagenet_dataset import UnlabeledImagenet
from pl_bolts.models.self_supervised.amdim import transforms as amdim_transforms
from pl_bolts.models.self_supervised.amdim.ssl_datasets import CIFAR10Mixed


class AMDIMPretraining():
    """"
    For pretraining we use the train transform for both train and val.
    """

    @staticmethod
    def cifar10(dataset_root, split: str = 'train'):
        assert split in ('train', 'val')
        dataset = CIFAR10Mixed(
            root=dataset_root,
            split=split,
            transform=amdim_transforms.AMDIMTrainTransformsCIFAR10(),
            download=True,
        )
        return dataset

    @staticmethod
    def cifar10_tiny(dataset_root, split: str = 'train'):
        assert split in ('train', 'val')
        dataset = CIFAR10Mixed(
            root=dataset_root,
            split=split,
            transform=amdim_transforms.AMDIMTrainTransformsCIFAR10(),
            download=True,
            nb_labeled_per_class=50,
        )
        return dataset

    @staticmethod
    def imagenet(dataset_root, nb_classes, split: str = 'train'):
        assert split in ('train', 'val')
        dataset = UnlabeledImagenet(
            dataset_root,
            nb_classes=nb_classes,
            split=split,
            transform=amdim_transforms.AMDIMTrainTransformsImageNet128(),
        )
        return dataset

    @staticmethod
    def stl(dataset_root, split: str = None):
        dataset = STL10(
            root=dataset_root,
            split='unlabeled',
            transform=amdim_transforms.AMDIMTrainTransformsSTL10(),
            download=True)
        tng_split, val_split = random_split(dataset, [95000, 5000])
        return tng_split, val_split

    @staticmethod
    def get_dataset(datamodule: str, data_dir, split: str = 'train', **kwargs):
        datasets = {
            'tiny-cifar10': AMDIMPretraining.cifar10_tiny,
            'cifar10': AMDIMPretraining.cifar10,
            'stl10': AMDIMPretraining.stl,
            'imagenet2012': AMDIMPretraining.imagenet,
        }
        assert datamodule in datasets, 'unrecognized dataset request'
        return datasets[datamodule](dataset_root=data_dir, split=split, **kwargs)


class AMDIMPatchesPretraining():
    """"
    For pretraining we use the train transform for both train and val.
    """

    @staticmethod
    def cifar10(dataset_root, patch_size, patch_overlap, split: str = 'train'):
        assert split in ('train', 'val')
        train_transform = amdim_transforms.TransformsC10Patches(
            patch_size=patch_size,
            patch_overlap=patch_overlap)
        dataset = CIFAR10Mixed(
            root=dataset_root,
            split=split,
            transform=train_transform,
            download=True,
        )
        return dataset

    @staticmethod
    def stl(dataset_root, patch_size, patch_overlap, split: str = None):
        train_transform = amdim_transforms.TransformsSTL10Patches(
            patch_size=patch_size,
            overlap=patch_overlap
        )
        dataset = STL10(
            root=dataset_root,
            split='unlabeled',
            transform=train_transform,
            download=True,
        )
        tng_split, val_split = random_split(dataset, [95000, 5000])

        return tng_split, val_split

    @staticmethod
    def imagenet(dataset_root, nb_classes, patch_size, patch_overlap, split: str = 'train'):
        assert split in ('train', 'val')
        train_transform = amdim_transforms.TransformsImageNet128Patches(
            patch_size=patch_size,
            overlap=patch_overlap
        )
        dataset = UnlabeledImagenet(
            dataset_root,
            nb_classes=nb_classes,
            split=split,
            transform=train_transform,
        )
        return dataset
