import random

from PIL import ImageFilter
from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import \
    imagenet_normalization, cifar10_normalization, stl10_normalization


class Moco2CIFAR10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization()
        ])
        self.finetune_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(44),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            cifar10_normalization(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2STL10Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            stl10_normalization()
        ])
        self.finetune_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            stl10_normalization(),
        ])
        self.test_augmentation = transforms.Compose([
            transforms.Resize(75),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            stl10_normalization(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2Imagenet128Transforms:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization()
        ])
        self.finetune_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
