from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class SimCLRTrainDataTransform:
    """Transforms for SimCLR during training step of the pre-training stage.

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        RandomApply([GaussianBlur(kernel_size=int(0.1 * self.input_height))], p=0.5)
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj, xk) = transform(x) # xk is only for the online evaluator if used
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([self.data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose(
            [transforms.RandomResizedCrop(self.input_height), transforms.RandomHorizontalFlip(), self.final_transform]
        )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """Transforms for SimCLR during the validation step of the pre-training stage.

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj, xk) = transform(x) # xk is only for the online evaluator if used
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ):
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class SimCLRFinetuneTransform(SimCLRTrainDataTransform):
    """Transforms for SimCLR during the fine-tuning stage.

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        xk = transform(x)
    """

    def __init__(
        self, input_height: int = 224, jitter_strength: float = 1.0, normalize=None, eval_transform: bool = False
    ) -> None:

        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=None, jitter_strength=jitter_strength
        )

        if eval_transform:
            self.data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
            ]

        self.transform = transforms.Compose([self.data_transforms, self.final_transform])

    def __call__(self, sample):
        return self.transform(sample)
