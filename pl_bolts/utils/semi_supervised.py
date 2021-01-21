import math
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from pl_bolts.utils import _SKLEARN_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _SKLEARN_AVAILABLE:
    from sklearn.utils import shuffle as sk_shuffle
else:  # pragma: no cover
    warn_missing_pkg('sklearn', pypi_name='scikit-learn')


class Identity(torch.nn.Module):
    """
    An identity class to replace arbitrary layers in pretrained models

    Example::

        from pl_bolts.utils import Identity

        model = resnet18()
        model.fc = Identity()

    """

    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


def balance_classes(X: Union[Tensor, np.ndarray], Y: Union[Tensor, np.ndarray, Sequence[int]],
                    batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes sure each batch has an equal amount of data from each class.
    Perfect balance

    Args:
        X: input features
        Y: mixed labels (ints)
        batch_size: the ultimate batch size
    """
    if not _SKLEARN_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError('You want to use `shuffle` function from `scikit-learn` which is not installed yet.')

    nb_classes = len(set(Y))

    nb_batches = math.ceil(len(Y) / batch_size)

    # sort by classes
    final_batches_x: List[list] = [[] for i in range(nb_batches)]
    final_batches_y: List[list] = [[] for i in range(nb_batches)]

    # Y needs to be np arr
    Y = np.asarray(Y)

    # pick chunk size for each class using the largest split
    chunk_sizes = []
    for class_i in range(nb_classes):
        mask = Y == class_i
        y = Y[mask]
        chunk_sizes.append(math.ceil(len(y) / nb_batches))
    chunk_size = max(chunk_sizes)
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
                final_batches_x[batch_i].append(x[i_start:i_end])
                final_batches_y[batch_i].append(y[i_start:i_end])

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
) -> Tuple[np.ndarray, np.ndarray]:
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

        X_larger = larger_set_X[i_start:i_end]
        Y_larger = larger_set_Y[i_start:i_end]

        # pull out labeled part
        smaller_start = i_start % (n_smaller - half_batch)
        smaller_end = smaller_start + half_batch

        X_small = smaller_set_X[smaller_start:smaller_end]
        Y_small = smaller_set_Y[smaller_start:smaller_end]

        X.extend([X_larger, X_small])
        Y.extend([Y_larger, Y_small])

    # aggregate reshuffled at end of shuffling
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y
