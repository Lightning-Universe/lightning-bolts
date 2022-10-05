import os
from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")

KITTI_LABELS = tuple(range(-1, 34))
DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)


class KittiDataset(Dataset):
    """KITTI Dataset for sematic segmentation.

    You need to have downloaded the Kitti semantic dataset first and provide the path to where it is saved.
    You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes, however not all of them are useful for training (e.g. railings on highways).
    Useful classes (the pixel values of these classes) are stored in `valid_labels`, other labels
    except useful classes are stored in `void_labels`.

    The class id and valid labels(`ignoreInEval`) can be found in here:
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

    Args:
        data_dir (str): where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
        img_size (tuple): image dimensions (width, height)
        valid_labels (tuple): useful classes to include
        transform (callable, optional): A function/transform that takes in the numpy array and transforms it.
    """

    IMAGE_PATH = os.path.join("training", "image_2")
    MASK_PATH = os.path.join("training", "semantic")

    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (1242, 376),
        valid_labels: Tuple[int] = DEFAULT_VALID_LABELS,
        transform: Optional[Callable] = None,
    ):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

        self.img_size = img_size
        self.valid_labels = valid_labels
        self.void_labels = tuple(label for label in KITTI_LABELS if label not in self.valid_labels)
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_dir, self.MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx])
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def encode_segmap(self, mask):
        """Sets all pixels of the mask with any of the `void_labels` to `ignore_index` (250 by default).

        It also sets all of the valid pixels to the appropriate value between 0 and `len(valid_labels)` (the number of
        valid classes), so it can be used properly by the loss function when comparing with the output.
        """
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        mask[mask > 33] = self.ignore_index
        return mask

    def get_filenames(self, path: str):
        """Returns a list of absolute paths to images inside given `path`"""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list
