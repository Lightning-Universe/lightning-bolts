import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from typing import Optional, List


class SwAVTrainDataTransform(object):
    def __init__(
        self,
        normalize: transforms.Normalize,
        size_crops: List[int] = [224, 96],
        nmb_crops: List[int] = [2, 6],
        min_scale_crops: List[float] = [0.14, 0.05],
        max_scale_crops: List[float] = [1., 0.14],
        gaussian_blur: bool = True,
        jitter_strength: float = 1.
    ):
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        transform = []
        color_transform = [
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            color_transform.append(
                GaussianBlur(kernel_size=int(0.1 * self.size_crops[0]), p=0.5)
            )

        self.color_transform = transforms.Compose(color_transform)

        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend([transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                transforms.ToTensor(),
                normalize])
            ] * self.nmb_crops[i])

        self.transform = transform

    def __call__(self, sample):
        multi_crops = list(
            map(lambda transform: transform(sample), self.transform)
        )

        return multi_crops


class SwAVEvalDataTransform(object):
    def __init__(
        self,
        normalize: transforms.Normalize,
        size_crops: List[int] = [224, 96],
        nmb_crops: List[int] = [2, 6],
        min_scale_crops: List[float] = [0.14, 0.05],
        max_scale_crops: List[float] = [1., 0.14],
        gaussian_blur: bool = True,
        jitter_strength: float = 1.
    ):
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        transform = []
        color_transform = [
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            color_transform.append(
                GaussianBlur(kernel_size=int(0.1 * self.size_crops[0]), p=0.5)
            )

        self.color_transform = transforms.Compose(color_transform)

        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend([transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                transforms.ToTensor(),
                normalize])
            ] * self.nmb_crops[i])

        self.transform = transform

    def __call__(self, sample):
        multi_crops = list(
            map(lambda transform: transform(sample), self.transform)
        )

        return multi_crops


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
