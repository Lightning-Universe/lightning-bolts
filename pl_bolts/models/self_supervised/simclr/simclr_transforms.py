import cv2
import numpy as np
import torchvision.transforms as transforms


class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """
    def __init__(self, input_height, s=1):
        self.s = s
        self.input_height = input_height
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_height),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.5),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_height)),
                                              transforms.ToTensor()])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class SimCLREvalDataTransform(object):
    """
    Transforms for SimCLR

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """
    def __init__(self, input_height, s=1):
        self.s = s
        self.input_height = input_height
        self.test_transform = transforms.Compose([
            transforms.Resize(input_height + 10, interpolation=3),
            transforms.CenterCrop(input_height),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        transform = self.test_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
