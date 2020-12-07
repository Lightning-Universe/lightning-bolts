import os

from pytorch_lightning import seed_everything

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
DATASETS_PATH = os.path.join(PACKAGE_ROOT, 'datasets')
# generate a list of random seeds for each test
ROOT_SEED = 1234

optional_pkg_names = [
    "torchvision",
    "gym",
    "sklearn",
    "PIL",
    "cv2",
]


def reset_seed():
    seed_everything()
