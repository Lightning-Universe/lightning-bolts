from typing import Tuple

import numpy as np

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg("cv2", pypi_name="opencv-python")


class SwAVTrainDataTransform:
    def __init__(
        self,
        normalize=None,
        size_crops: Tuple[int] = (96, 36),
        nmb_crops: Tuple[int] = (2, 4),
        min_scale_crops: Tuple[float] = (0.33, 0.10),
        max_scale_crops: Tuple[float] = (1, 0.33),
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
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
            0.2 * self.jitter_strength,
        )

        transform = []
        color_transform = [transforms.RandomApply([self.color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.size_crops[0])
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend(
                [
                    transforms.Compose(
                        [
                            random_resized_crop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            self.color_transform,
                            self.final_transform,
                        ]
                    )
                ]
                * self.nmb_crops[i]
            )

        self.transform = transform

        # add online train transform of the size of global view
        online_train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(self.size_crops[0]), transforms.RandomHorizontalFlip(), self.final_transform]
        )

        self.transform.append(online_train_transform)

    def __call__(self, sample):
        multi_crops = list(map(lambda transform: transform(sample), self.transform))

        return multi_crops


class SwAVEvalDataTransform(SwAVTrainDataTransform):
    def __init__(
        self,
        normalize=None,
        size_crops: Tuple[int] = (96, 36),
        nmb_crops: Tuple[int] = (2, 4),
        min_scale_crops: Tuple[float] = (0.33, 0.10),
        max_scale_crops: Tuple[float] = (1, 0.33),
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
    ):
        super().__init__(
            normalize=normalize,
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
        )

        input_height = self.size_crops[0]  # get global view crop
        test_transform = transforms.Compose(
            [
                transforms.Resize(int(input_height + 0.1 * input_height)),
                transforms.CenterCrop(input_height),
                self.final_transform,
            ]
        )

        # replace last transform to eval transform in self.transform list
        self.transform[-1] = test_transform


class SwAVFinetuneTransform:
    def __init__(
        self, input_height: int = 224, jitter_strength: float = 1.0, normalize=None, eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur:
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
