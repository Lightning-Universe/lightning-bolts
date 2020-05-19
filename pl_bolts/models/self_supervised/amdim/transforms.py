from torchvision import transforms
from pl_bolts.transforms.self_supervised import RandomTranslateWithReflect


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsSTL10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
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

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(70, interpolation=3),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize
            ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsImageNet128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
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
        out2 = self.train_transform(inp)
        return out1, out2
