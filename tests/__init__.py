import os

import torch
from lightning.pytorch import seed_everything

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(TEST_ROOT)
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(PROJECT_ROOT, "_datasets"))
# generate a list of random seeds for each test
ROOT_SEED = 1234

_MARK_REQUIRE_GPU = {"condition": not torch.cuda.is_available(), "reason": "test requires GPU machine"}


def reset_seed():
    seed_everything()
