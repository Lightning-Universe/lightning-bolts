import os

import numpy as np
import torch

TEST_ROOT = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
# generate a list of random seeds for each test
ROOT_SEED = 1234


def reset_seed():
    torch.manual_seed(ROOT_SEED)
    np.random.seed(ROOT_SEED)
