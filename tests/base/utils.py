import numpy as np
import torch

RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))


def reset_seed():
    seed = RANDOM_SEEDS.pop()
    torch.manual_seed(seed)
    np.random.seed(seed)
