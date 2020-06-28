import math
from abc import ABC
from typing import Callable

import numpy as np
from sklearn.utils import shuffle
from torchvision.datasets import CIFAR10


class SSLDatasetMixin(ABC):

    @classmethod
    def generate_train_val_split(cls, examples, labels, pct_val):
        """
        Splits dataset uniformly across classes
        :param examples:
        :param labels:
        :param pct_val:
        :return:
        """
        nb_classes = len(set(labels))

        nb_val_images = int(len(examples) * pct_val) // nb_classes

        val_x = []
        val_y = []
        train_x = []
        train_y = []

        cts = {x: 0 for x in range(nb_classes)}
        for img, class_idx in zip(examples, labels):

            # allow labeled
            if cts[class_idx] < nb_val_images:
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
    def select_nb_imgs_per_class(cls, examples, labels, nb_imgs_in_val):
        """
        Splits a dataset into two parts.
        The labeled split has nb_imgs_in_val per class
        :param examples:
        :param labels:
        :param nb_imgs_in_val:
        :return:
        """
        nb_classes = len(set(labels))

        # def partition_train_set(self, imgs, nb_imgs_in_val):
        labeled = []
        labeled_y = []
        unlabeled = []
        unlabeled_y = []

        cts = {x: 0 for x in range(nb_classes)}
        for img_name, class_idx in zip(examples, labels):

            # allow labeled
            if cts[class_idx] < nb_imgs_in_val:
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
        idxs = shuffle(idxs, random_state=1234)

        x = x[idxs]

        y = np.asarray(y)
        y = y[idxs]
        y = list(y)

        return x, y


class CIFAR10Mixed(SSLDatasetMixin, CIFAR10):

    def __init__(
            self,
            root: str,
            split: str = 'val',
            transform: Callable = None,
            target_transform: Callable = None,
            download: bool = False,
            nb_labeled_per_class: int = None,
            val_pct: float = 0.10
    ):

        if nb_labeled_per_class == -1:
            nb_labeled_per_class = None

        # use train for all of these splits
        train = split in ('val', 'train', 'train+unlabeled')
        super().__init__(root, train, transform, target_transform, download)

        # modify only for val, train
        if split != 'test':
            # limit nb of examples per class
            X_test, y_test, X_train, y_train = self.generate_train_val_split(self.data, self.targets, val_pct)

            # shuffle idxs representing the data
            X_train, y_train = self.deterministic_shuffle(X_train, y_train)
            X_test, y_test = self.deterministic_shuffle(X_test, y_test)

            if split == 'val':
                self.data = X_test
                self.targets = y_test

            else:
                self.data = X_train
                self.targets = y_train

            # limit the number of items per class
            if nb_labeled_per_class is not None:
                self.data, self.targets = \
                    self.select_nb_imgs_per_class(self.data, self.targets, nb_labeled_per_class)
