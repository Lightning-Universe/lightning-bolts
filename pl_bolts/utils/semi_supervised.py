import torch
import numpy as np
import math
from sklearn.utils import shuffle as sk_shuffle


class Identity(torch.nn.Module):
    """
    An identity class to replace arbitrary layers in pretrained models

    Example::

        from pl_bolts.utils import Identity

        model = resnet18()
        model.fc = Identity()

    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def balance_classes(X: np.ndarray, Y: list, batch_size: int):
    """
    Makes sure each batch has an equal amount of data from each class.
    Perfect balance

    Args:

        X: input features
        Y: mixed labels (ints)
        batch_size: the ultimate batch size
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
        x, y = sk_shuffle(x, y, random_state=123)

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


def generate_half_labeled_batches(
        smaller_set_X: np.ndarray,
        smaller_set_Y: np.ndarray,
        larger_set_X: np.ndarray,
        larger_set_Y: np.ndarray,
        batch_size: int,
):
    """
    Given a labeled dataset and an unlabeled dataset, this function generates
    a joint pair where half the batches are labeled and the other half is not

    """
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
