import numpy as np
import os
import torch

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))
ROOT_SEED = 1234
torch.manual_seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


def reset_seed():
    seed = RANDOM_SEEDS.pop()
    torch.manual_seed(seed)
    np.random.seed(seed)
