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
    def cifar10_train(dataset_root):
        train_transform = amdim_transforms.TransformsC10()
        dataset = CIFAR10Mixed(root=dataset_root, split='train', transform=train_transform, download=True)

        return dataset

    @staticmethod
    def stl_train(dataset_root):
        train_transform = amdim_transforms.TransformsSTL10()
        dataset = STL10(root=dataset_root, split='unlabeled', transform=train_transform, download=True)
        tng_split, val_split = random_split(dataset, [95000, 5000])

        return tng_split, val_split

    @staticmethod
    def imagenet_train(dataset_root, nb_classes):
        train_transform = amdim_transforms.TransformsImageNet128()
        dataset = UnlabeledImagenet(dataset_root,
                                    nb_classes=nb_classes,
                                    split='train',
                                    transform=train_transform)
        return dataset

    @staticmethod
    def cifar10_val(dataset_root):
        train_transform = amdim_transforms.TransformsC10()
        dataset = CIFAR10Mixed(root=dataset_root, split='val', transform=train_transform, download=True)

        return dataset

    @staticmethod
    def stl_val(dataset_root):
        # VAL COMES FROM CALLING STL_TRAIN
        return None

    @staticmethod
    def imagenet_val(dataset_root, nb_classes):
        train_transform = amdim_transforms.TransformsImageNet128()
        dataset = UnlabeledImagenet(dataset_root,
                                    nb_classes=nb_classes,
                                    split='val',
                                    transform=train_transform)
        return dataset


class AMDIMPatchesPretraining():
    """"
    For pretraining we use the train transform for both train and val.
    """
    @staticmethod
    def cifar10_train(dataset_root, patch_size, patch_overlap):
        train_transform = amdim_transforms.TransformsC10Patches(
            patch_size=patch_size,
            patch_overlap=patch_overlap)
        dataset = CIFAR10Mixed(root=dataset_root, split='train', transform=train_transform, download=True)

        return dataset

    @staticmethod
    def stl_train(dataset_root, patch_size, patch_overlap):
        train_transform = amdim_transforms.TransformsSTL10Patches(
            patch_size=patch_size,
            overlap=patch_overlap
        )
        dataset = STL10(root=dataset_root, split='unlabeled', transform=train_transform, download=True)
        tng_split, val_split = random_split(dataset, [95000, 5000])

        return tng_split, val_split

    @staticmethod
    def imagenet_train(dataset_root, nb_classes, patch_size, patch_overlap):
        train_transform = amdim_transforms.TransformsImageNet128Patches(
            patch_size=patch_size,
            overlap=patch_overlap
        )
        dataset = UnlabeledImagenet(dataset_root,
                                    nb_classes=nb_classes,
                                    split='train',
                                    transform=train_transform)
        return dataset

    @staticmethod
    def cifar10_val(dataset_root, patch_size, patch_overlap):
        train_transform = amdim_transforms.TransformsC10Patches(
            patch_size=patch_size,
            patch_overlap=patch_overlap)
        dataset = CIFAR10Mixed(root=dataset_root, split='val', transform=train_transform, download=True)

        return dataset

    @staticmethod
    def stl_val(dataset_root):
        # VAL COMES FROM CALLING STL_TRAIN
        return None

    @staticmethod
    def imagenet_val(dataset_root, nb_classes, patch_size, patch_overlap):
        train_transform = amdim_transforms.TransformsImageNet128Patches(
            patch_size=patch_size,
            overlap=patch_overlap
        )
        dataset = UnlabeledImagenet(dataset_root,
                                    nb_classes=nb_classes,
                                    split='val',
                                    transform=train_transform)
        return dataset
