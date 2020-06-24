from torchvision import transforms

from pl_bolts.transforms.self_supervised import RandomTranslateWithReflect, Patchify


class CPCTrainTransformsCIFAR10:

    def __init__(self, patch_size=8):
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
        """
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)

        self.transforms = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=patch_size // 2),
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCEvalTransformsCIFAR10:

    def __init__(self, patch_size=8):
        """
        Transforms used for CPC:

        Args:
            patch_size: size of patches when cutting up the image into overlapping patches

        Transforms::

            random_flip
            transforms.ToTensor()
            normalize
            Patchify(patch_size=patch_size, overlap_size=patch_size // 2)
        """

        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=patch_size // 2),
        ])

    def __call__(self, inp):
        out1 = self.transforms(inp)
        return out1


class CPCTransformsSTL10Patches:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self, patch_size, overlap):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        # image augmentation functions
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = \
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)

        self.test_transform = transforms.Compose([
            transforms.Resize(70, interpolation=3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize,
            Patchify(patch_size=patch_size, overlap_size=overlap)
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        return out1


class CPCTransformsImageNet128Patches:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''

    def __init__(self, patch_size, overlap):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        return out1
