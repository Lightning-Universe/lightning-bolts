import numpy as np
from torchvision.datasets import CIFAR10, VisionDataset
from sklearn.utils import shuffle
import math


class SSLDatasetMixin(VisionDataset):
    def generate_train_val_split(self, examples, labels, pct_val):
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


    def select_nb_imgs_per_class(self, examples, labels, nb_imgs_in_val):
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

    def deterministic_shuffle(self, x, y):
        n = len(x)
        idxs = list(range(0, n))
        idxs = shuffle(idxs, random_state=1234)

        x = x[idxs]

        y = np.asarray(y)
        y = y[idxs]
        y = list(y)

        return x, y

class CIFAR10Mixed(SSLDatasetMixin, CIFAR10):

    def __init__(self, root, split='val',
                 transform=None, target_transform=None,
                 download=False, nb_labeled_per_class=None, val_pct=0.10):

        if nb_labeled_per_class == -1:
            nb_labeled_per_class = None

        # use train for all of these splits
        train = split == 'val' or split == 'train' or split == 'train+unlabeled'
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
                self.data, self.targets = self.select_nb_imgs_per_class(self.data,
                                                                        self.targets,
                                                                        nb_labeled_per_class)

    def __balance_class_batches(self, X, Y, batch_size):
        """
        Makes sure each batch has an equal amount of data from each class.
        Perfect balance
        :param X:
        :param Y:
        :param batch_size:
        :return:
        """

        nb_classes = len(set(Y))
        nb_batches = math.ceil(len(Y) / batch_size)

        # sort by classes
        final_batches_x = [[] for i in range(nb_batches)]
        final_batches_y = [[] for i in range(nb_batches)]

        # Y needs to be np arr
        Y = np.asarray(Y)

        # pick chunk size for each class using the largest split
        chunk_size = []
        for class_i in range(nb_classes):
            mask = Y == class_i
            y = Y[mask]
            chunk_size.append(math.ceil(len(y) / nb_batches))
        chunk_size = max(chunk_size)
        # force chunk size to be even
        if chunk_size % 2 != 0:
            chunk_size -= 1

        # divide each class into each batch
        for class_i in range(nb_classes):
            mask = Y == class_i
            x = X[mask]
            y = Y[mask]

            # shuffle items in the class
            x, y = shuffle(x, y, random_state=123)

            # divide the class into the batches
            for i_start in range(0, len(y), chunk_size):
                batch_i = i_start // chunk_size
                i_end = i_start + chunk_size

                if len(final_batches_x) > batch_i:
                    final_batches_x[batch_i].append(x[i_start: i_end])
                    final_batches_y[batch_i].append(y[i_start: i_end])

        # merge into full dataset
        final_batches_x = [np.concatenate(x, axis=0) for x in final_batches_x if len(x) > 0]
        final_batches_x = np.concatenate(final_batches_x, axis=0)

        final_batches_y = [np.concatenate(x, axis=0) for x in final_batches_y if len(x) > 0]
        final_batches_y = np.concatenate(final_batches_y, axis=0)

        return final_batches_x, final_batches_y


    def __generate_half_labeled_batches(self, smaller_set_X, smaller_set_Y,
                                        larger_set_X, larger_set_Y,
                                        batch_size):
        X = []
        Y = []
        half_batch = batch_size // 2

        n_larger = len(larger_set_X)
        n_smaller = len(smaller_set_X)
        for i_start in range(0, n_larger, half_batch):
            i_end = i_start + half_batch

            X_larger = larger_set_X[i_start: i_end]
            Y_larger = larger_set_Y[i_start: i_end]

            # pull out labeled part
            smaller_start = i_start % (n_smaller - half_batch)
            smaller_end = smaller_start + half_batch

            X_small = smaller_set_X[smaller_start: smaller_end]
            Y_small = smaller_set_Y[smaller_start:smaller_end]

            X.extend([X_larger, X_small])
            Y.extend([Y_larger, Y_small])

        # aggregate reshuffled at end of shuffling
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        return X, Y