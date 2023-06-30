from typing import List, Tuple

from torch import Tensor

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class SwAVTrainDataTransform:
    def __init__(
        self,
        normalize=None,
        size_crops: Tuple[int] = (96, 36),
        num_crops: Tuple[int] = (2, 4),
        min_scale_crops: Tuple[float] = (0.33, 0.10),
        max_scale_crops: Tuple[float] = (1, 0.33),
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
    ) -> None:
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur

        if len(size_crops) != len(num_crops):
            raise AssertionError("len(size_crops) should equal len(num_crops).")
        if len(min_scale_crops) != len(num_crops):
            raise AssertionError("len(min_scale_crops) should equal len(num_crops).")
        if len(max_scale_crops) != len(num_crops):
            raise AssertionError("len(max_scale_crops) should equal len(num_crops).")

        self.size_crops = size_crops
        self.num_crops = num_crops
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

            # Resort to torchvision gaussian blur instead of custom implementation
            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

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
                * self.num_crops[i]
            )

        self.transform = transform

        # add online train transform of the size of global view
        online_train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(self.size_crops[0]), transforms.RandomHorizontalFlip(), self.final_transform]
        )

        self.transform.append(online_train_transform)

    def __call__(self, sample: Tensor) -> List[Tensor]:
        return [transform(sample) for transform in self.transform]


class SwAVEvalDataTransform(SwAVTrainDataTransform):
    def __init__(
        self,
        normalize=None,
        size_crops: Tuple[int] = (96, 36),
        num_crops: Tuple[int] = (2, 4),
        min_scale_crops: Tuple[float] = (0.33, 0.10),
        max_scale_crops: Tuple[float] = (1, 0.33),
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
    ) -> None:
        super().__init__(
            normalize=normalize,
            size_crops=size_crops,
            num_crops=num_crops,
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

    def __call__(self, sample: Tensor) -> Tensor:
        return self.transform(sample)
