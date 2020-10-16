import os

from pytorch_lightning import seed_everything

TEST_ROOT = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
# generate a list of random seeds for each test
ROOT_SEED = 1234


def reset_seed():
    seed_everything()
