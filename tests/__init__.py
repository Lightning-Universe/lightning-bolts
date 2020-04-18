import os

import numpy as np
import torch

# generate a list of random seeds for each test
ROOT_SEED = 1234


def reset_seed():
    torch.manual_seed(ROOT_SEED)
    np.random.seed(ROOT_SEED)
