import gzip
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager

import numpy as np
import torch

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

PY3 = sys.version_info[0] == 3

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import ImageNet
    from torchvision.datasets.imagenet import load_meta_file
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    ImageNet = object


class UnlabeledImagenet(ImageNet):
    """Official train set gets split into train, val. (using nb_imgs_per_val_class for each class). Official
    validation becomes test set.

    Within each class, we further allow limiting the number of samples per class (for semi-sup lng)
    """

    def __init__(
        self,
        root,
        split: str = "train",
        num_classes: int = -1,
        num_imgs_per_class: int = -1,
        num_imgs_per_class_val_split: int = 50,
        meta_dir=None,
        **kwargs,
    ):
        """
        Args:
            root: path of dataset
            split:
            num_classes: Sets the limit of classes
            num_imgs_per_class: Limits the number of images per class
            num_imgs_per_class_val_split: How many images per class to generate the val split
            download:
            kwargs:
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
            )

        root = self.root = os.path.expanduser(root)

        # [train], [val] --> [train, val], [test]
        original_split = split
        if split == "train" or split == "val":
            split = "train"

        if split == "test":
            split = "val"

        self.split = split
        split_root = os.path.join(root, split)
        meta_dir = meta_dir if meta_dir is not None else split_root
        wnid_to_classes = load_meta_file(meta_dir)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        # shuffle images first
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

        # partition train set into [train, val]
        if split == "train":
            train, val = self.partition_train_set(self.imgs, num_imgs_per_class_val_split)
            if original_split == "train":
                self.imgs = train
            if original_split == "val":
                self.imgs = val

        # limit the number of images in train or test set since the limit was already applied to the val set
        if split in ["train", "test"]:
            if num_imgs_per_class != -1:
                clean_imgs = []
                cts = {x: 0 for x in range(len(self.classes))}
                for img_name, idx in self.imgs:
                    if cts[idx] < num_imgs_per_class:
                        clean_imgs.append((img_name, idx))
                        cts[idx] += 1

                self.imgs = clean_imgs

        # limit the number of classes
        if num_classes != -1:
            # choose the classes at random (but deterministic)
            ok_classes = list(range(num_classes))
            np.random.seed(1234)
            np.random.shuffle(ok_classes)
            ok_classes = ok_classes[:num_classes]
            ok_classes = set(ok_classes)

            clean_imgs = []
            for img_name, idx in self.imgs:
                if idx in ok_classes:
                    clean_imgs.append((img_name, idx))

            self.imgs = clean_imgs

        # shuffle again for final exit
        np.random.seed(1234)
        np.random.shuffle(self.imgs)

        # list of class_nbs for each image
        idcs = [idx for _, idx in self.imgs]

        self.wnids = self.classes
        self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for clss, idx in zip(self.classes, idcs) for cls in clss}

        # update the root data
        self.samples = self.imgs
        self.targets = [s[1] for s in self.imgs]

    def partition_train_set(self, imgs, nb_imgs_in_val):
        val = []
        train = []

        cts = {x: 0 for x in range(len(self.classes))}
        for img_name, idx in imgs:
            if cts[idx] < nb_imgs_in_val:
                val.append((img_name, idx))
                cts[idx] += 1
            else:
                train.append((img_name, idx))

        return train, val

    @classmethod
    def generate_meta_bins(cls, devkit_dir):
        files = os.listdir(devkit_dir)
        if "ILSVRC2012_devkit_t12.tar.gz" not in files:
            raise FileNotFoundError(
                "devkit_path must point to the devkit file"
                "ILSVRC2012_devkit_t12.tar.gz. Download from here:"
                "http://www.image-net.org/challenges/LSVRC/2012/downloads"
            )

        parse_devkit_archive(devkit_dir)
        print(f"meta.bin generated at {devkit_dir}/meta.bin")


def _verify_archive(root, file, md5):
    if not _check_integrity(os.path.join(root, file), md5):
        raise RuntimeError(
            f"The archive {file} is not present in the root directory or is corrupted."
            f" You need to download it externally and place it in {root}."
        )


def _check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return _check_md5(fpath, md5)


def _check_md5(fpath, md5, **kwargs):
    return md5 == _calculate_md5(fpath, **kwargs)


def _calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def parse_devkit_archive(root, file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save the meta information in a
    binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    from scipy import io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir():
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf")
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        META_FILE = "meta.bin"

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    PY3 = sys.version_info[0] == 3

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_zip(filename):
    return filename.endswith(".zip")
