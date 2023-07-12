from abc import ABC
from typing import Callable, Optional

import numpy as np

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import CIFAR10
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    CIFAR10 = object


@under_review()
class SSLDatasetMixin(ABC):
    @classmethod
    def generate_train_val_split(cls, examples, labels, pct_val):
        """Splits dataset uniformly across classes."""
        num_classes = len(set(labels))

        num_val_images = int(len(examples) * pct_val) // num_classes

        val_x = []
        val_y = []
        train_x = []
        train_y = []

        cts = {x: 0 for x in range(num_classes)}
        for img, class_idx in zip(examples, labels):
            # allow labeled
            if cts[class_idx] < num_val_images:
                val_x.append(img)
                val_y.append(class_idx)
                cts[class_idx] += 1
            else:
                train_x.append(img)
                train_y.append(class_idx)

        val_x = np.stack(val_x)
        train_x = np.stack(train_x)
        return val_x, val_y, train_x, train_y

    @classmethod
    def select_num_imgs_per_class(cls, examples, labels, num_imgs_in_val):
        """Splits a dataset into two parts.

        The labeled split has num_imgs_in_val per class

        """
        num_classes = len(set(labels))

        # def partition_train_set(self, imgs, num_imgs_in_val):
        labeled = []
        labeled_y = []
        unlabeled = []
        unlabeled_y = []

        cts = {x: 0 for x in range(num_classes)}
        for img_name, class_idx in zip(examples, labels):
            # allow labeled
            if cts[class_idx] < num_imgs_in_val:
                labeled.append(img_name)
                labeled_y.append(class_idx)
                cts[class_idx] += 1
            else:
                unlabeled.append(img_name)
                unlabeled_y.append(class_idx)

        labeled = np.stack(labeled)

        return labeled, labeled_y

    @classmethod
    def deterministic_shuffle(cls, x, y):
        n = len(x)
        idxs = list(range(0, n))
        np.random.seed(1234)
        np.random.shuffle(idxs)

        x = x[idxs]

        y = np.asarray(y)
        y = y[idxs]
        y = list(y)

        return x, y


@under_review()
class CIFAR10Mixed(SSLDatasetMixin, CIFAR10):
    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_labeled_per_class: Optional[int] = None,
        val_pct: float = 0.10,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        if num_labeled_per_class == -1:
            num_labeled_per_class = None

        # use train for all of these splits
        train = split in ("val", "train", "train+unlabeled")
        super(SSLDatasetMixin, self).__init__(root, train, transform, target_transform, download)

        # modify only for val, train
        if split != "test":
            # limit nb of examples per class
            data_test, lbs_test, data_train, lbs_train = self.generate_train_val_split(self.data, self.targets, val_pct)

            # shuffle idxs representing the data
            data_train, lbs_train = self.deterministic_shuffle(data_train, lbs_train)
            data_test, lbs_test = self.deterministic_shuffle(data_test, lbs_test)

            if split == "val":
                self.data = data_test
                self.targets = lbs_test

            else:
                self.data = data_train
                self.targets = lbs_train

            # limit the number of items per class
            if num_labeled_per_class is not None:
                self.data, self.targets = self.select_num_imgs_per_class(self.data, self.targets, num_labeled_per_class)
