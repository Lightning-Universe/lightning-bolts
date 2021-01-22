from pl_bolts.transforms.self_supervised import Patchify, RandomTranslateWithReflect
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class CPCTrainTransformsCIFAR10:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        img_jitter
        col_jitter
        rnd_gray
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

    Example::

        # in a regular dataset
        CIFAR10(..., transforms=CPCTrainTransformsCIFAR10())

        # in a DataModule
        module = CIFAR10DataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsCIFAR10())

    """

    def __init__(self, patch_size=8, overlap=4):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)

        self.transforms = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCEvalTransformsCIFAR10:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=overlap)

    Example::

        # in a regular dataset
        CIFAR10(..., transforms=CPCEvalTransformsCIFAR10())

        # in a DataModule
        module = CIFAR10DataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsCIFAR10())

    """

    def __init__(self, patch_size: int = 8, overlap: int = 4):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # flipping image along vertical axis
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])

    def __call__(self, inp):
        out1 = self.transforms(inp)
        return out1


class CPCTrainTransformsSTL10:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        img_jitter
        col_jitter
        rnd_gray
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

    Example::

        # in a regular dataset
        STL10(..., transforms=CPCTrainTransformsSTL10())

        # in a DataModule
        module = STL10DataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsSTL10())
    """

    def __init__(self, patch_size: int = 16, overlap: int = 8):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # flipping image along vertical axis
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))

        # image augmentation functions
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3)

        self.transforms = transforms.Compose([
            rand_crop, col_jitter, rnd_gray,
            transforms.ToTensor(), normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCEvalTransformsSTL10:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

    Example::

        # in a regular dataset
        STL10(..., transforms=CPCEvalTransformsSTL10())

        # in a DataModule
        module = STL10DataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsSTL10())

    """

    def __init__(self, patch_size: int = 16, overlap: int = 8):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # flipping image along vertical axis
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))

        self.transforms = transforms.Compose([
            transforms.Resize(70, interpolation=3),
            transforms.CenterCrop(64),
            transforms.ToTensor(), normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

    def __call__(self, inp):
        out1 = self.transforms(inp)
        return out1


class CPCTrainTransformsImageNet128:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

    Example::

        # in a regular dataset
        Imagenet(..., transforms=CPCTrainTransformsImageNet128())

        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsImageNet128())
    """

    def __init__(self, patch_size: int = 32, overlap: int = 16):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)

        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])

        self.transforms = transforms.Compose([rand_crop, col_jitter, rnd_gray, post_transform])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCEvalTransformsImageNet128:
    """
    Transforms used for CPC:

    Transforms::

        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)

    Example::

        # in a regular dataset
        Imagenet(..., transforms=CPCEvalTransformsImageNet128())

        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsImageNet128())
    """

    def __init__(self, patch_size: int = 32, overlap: int = 16):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.transforms = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128), post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1
