from collections import Counter

import numpy as np
import torch

from pl_bolts.utils.semi_supervised import balance_classes, generate_half_labeled_batches


def test_balance_classes():
    X = torch.rand(100, 3, 32, 32)
    c1 = torch.zeros(20, 1)
    c2 = torch.zeros(20, 1) + 1
    c3 = torch.zeros(60, 1) + 2

    y = torch.cat([c1, c2, c3], dim=0).int().cpu().numpy().flatten().tolist()
    (balanced_X, balanced_Y) = balance_classes(X, y, batch_size=10)


def test_generate_half_labeled_batches():
    smaller_set_X = np.random.rand(100, 3, 32, 32)
    smaller_set_Y = np.random.randint(0, 3, (100, 1))
    larger_set_X = np.random.rand(100, 3, 32, 32)
    larger_set_Y = np.zeros_like(smaller_set_Y) + -1

    (mixed_x, mixed_y) = generate_half_labeled_batches(smaller_set_X, smaller_set_Y, larger_set_X, larger_set_Y, 10)

    # only half the batch should be unlabeled
    for i in range(0, len(mixed_y), 10):
        batch = mixed_y[i : i + 10]
        counts = Counter(batch.flatten().tolist())
        assert counts[-1] == 5
