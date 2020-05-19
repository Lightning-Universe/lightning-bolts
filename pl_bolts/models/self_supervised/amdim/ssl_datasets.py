import gzip
import hashlib
import math
import os
import shutil
import tarfile
import tempfile
import zipfile
from abc import ABC
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.utils import shuffle
from torch._six import PY3
from torchvision.datasets import CIFAR10, VisionDataset, ImageNet
from torchvision.datasets.imagenet import load_meta_file


class SSLDatasetMixin(ABC):

    @classmethod
    def generate_train_val_split(cls, examples, labels, pct_val):
        """
        Splits dataset uniformly across classes
        :param examples:
        :param labels:
        :param pct_val:
        :return:
        """
        nb_classes = len(set(labels))

        nb_val_images = int(len(examples) * pct_val) // nb_classes

        val_x = []
        val_y = []
        train_x = []
        train_y = []

        cts = {x: 0 for x in range(nb_classes)}
        for img, class_idx in zip(examples, labels):

            # allow labeled
            if cts[class_idx] < nb_val_images:
                val_x.append(img)
                val_y.append(class_idx)
                cts[class_idx] += 1
            else:
                train_x.append(img)
                train_y.append(class_idx)

        val_x = np.stack(val_x)
        train_x = np.stack(train_x)
        return val_x, val_y, train_x, train_y

    @classmethod
    def select_nb_imgs_per_class(cls, examples, labels, nb_imgs_in_val):
        """
        Splits a dataset into two parts.
        The labeled split has nb_imgs_in_val per class
        :param examples:
        :param labels:
        :param nb_imgs_in_val:
        :return:
        """
        nb_classes = len(set(labels))

        # def partition_train_set(self, imgs, nb_imgs_in_val):
        labeled = []
        labeled_y = []
        unlabeled = []
        unlabeled_y = []

        cts = {x: 0 for x in range(nb_classes)}
        for img_name, class_idx in zip(examples, labels):

            # allow labeled
            if cts[class_idx] < nb_imgs_in_val:
                labeled.append(img_name)
                labeled_y.append(class_idx)
                cts[class_idx] += 1
            else:
                unlabeled.append(img_name)
                unlabeled_y.append(class_idx)

        labeled = np.stack(labeled)

        return labeled, labeled_y

    @classmethod
    def deterministic_shuffle(cls, x, y):
        n = len(x)
        idxs = list(range(0, n))
        idxs = shuffle(idxs, random_state=1234)

        x = x[idxs]

        y = np.asarray(y)
        y = y[idxs]
        y = list(y)

        return x, y


class CIFAR10Mixed(SSLDatasetMixin, CIFAR10):

    def __init__(self, root, split='val',
                 transform=None, target_transform=None,
                 download=False, nb_labeled_per_class=None, val_pct=0.10):

        if nb_labeled_per_class == -1:
            nb_labeled_per_class = None

        # use train for all of these splits
        train = split == 'val' or split == 'train' or split == 'train+unlabeled'
        super().__init__(root, train, transform, target_transform, download)

        # modify only for val, train
        if split != 'test':
            # limit nb of examples per class
            X_test, y_test, X_train, y_train = self.generate_train_val_split(self.data, self.targets, val_pct)

            # shuffle idxs representing the data
            X_train, y_train = self.deterministic_shuffle(X_train, y_train)
            X_test, y_test = self.deterministic_shuffle(X_test, y_test)

            if split == 'val':
                self.data = X_test
                self.targets = y_test

            else:
                self.data = X_train
                self.targets = y_train

            # limit the number of items per class
            if nb_labeled_per_class is not None:
                self.data, self.targets = self.select_nb_imgs_per_class(self.data,
                                                                        self.targets,
                                                                        nb_labeled_per_class)

    def __balance_class_batches(self, X, Y, batch_size):
        """
        Makes sure each batch has an equal amount of data from each class.
        Perfect balance
        :param X:
        :param Y:
        :param batch_size:
        :return:
        """

        nb_classes = len(set(Y))
        nb_batches = math.ceil(len(Y) / batch_size)

        # sort by classes
        final_batches_x = [[] for i in range(nb_batches)]
        final_batches_y = [[] for i in range(nb_batches)]

        # Y needs to be np arr
        Y = np.asarray(Y)

        # pick chunk size for each class using the largest split
        chunk_size = []
        for class_i in range(nb_classes):
            mask = Y == class_i
            y = Y[mask]
            chunk_size.append(math.ceil(len(y) / nb_batches))
        chunk_size = max(chunk_size)
        # force chunk size to be even
        if chunk_size % 2 != 0:
            chunk_size -= 1

        # divide each class into each batch
        for class_i in range(nb_classes):
            mask = Y == class_i
            x = X[mask]
            y = Y[mask]

            # shuffle items in the class
            x, y = shuffle(x, y, random_state=123)

            # divide the class into the batches
            for i_start in range(0, len(y), chunk_size):
                batch_i = i_start // chunk_size
                i_end = i_start + chunk_size

                if len(final_batches_x) > batch_i:
                    final_batches_x[batch_i].append(x[i_start: i_end])
                    final_batches_y[batch_i].append(y[i_start: i_end])

        # merge into full dataset
        final_batches_x = [np.concatenate(x, axis=0) for x in final_batches_x if len(x) > 0]
        final_batches_x = np.concatenate(final_batches_x, axis=0)

        final_batches_y = [np.concatenate(x, axis=0) for x in final_batches_y if len(x) > 0]
        final_batches_y = np.concatenate(final_batches_y, axis=0)

        return final_batches_x, final_batches_y

    def __generate_half_labeled_batches(self, smaller_set_X, smaller_set_Y,
                                        larger_set_X, larger_set_Y,
                                        batch_size):
        X = []
        Y = []
        half_batch = batch_size // 2

        n_larger = len(larger_set_X)
        n_smaller = len(smaller_set_X)
        for i_start in range(0, n_larger, half_batch):
            i_end = i_start + half_batch

            X_larger = larger_set_X[i_start: i_end]
            Y_larger = larger_set_Y[i_start: i_end]

            # pull out labeled part
            smaller_start = i_start % (n_smaller - half_batch)
            smaller_end = smaller_start + half_batch

            X_small = smaller_set_X[smaller_start: smaller_end]
            Y_small = smaller_set_Y[smaller_start:smaller_end]

            X.extend([X_larger, X_small])
            Y.extend([Y_larger, Y_small])

        # aggregate reshuffled at end of shuffling
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        return X, Y


class UnlabeledImagenet(ImageNet):

    def __init__(self, root, split='train',
                 num_classes=-1,
                 num_imgs_per_class=-1,
                 num_imgs_per_class_val_split=50,
                 **kwargs):
        """
        Official train set gets split into train, val. (using nb_imgs_per_val_class for each class).
        Official validation becomes test set

        Within each class, we further allow limiting the number of samples per class (for semi-sup lng)

        :param root: path of dataset
        :param split:
        :param num_classes: Sets the limit of classes
        :param num_imgs_per_class: Limits the number of images per class
        :param num_imgs_per_class_val_split: How many images per class to generate the val split
        :param download:
        :param kwargs:
        """
        root = self.root = os.path.expanduser(root)

        # [train], [val] --> [train, val], [test]
        original_split = split
        if split == 'train' or split == 'val':
            split = 'train'

        if split == 'test':
            split = 'val'

        self.split = split
        wnid_to_classes = load_meta_file(root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        # shuffle images first
        self.imgs = shuffle(self.imgs, random_state=1234)

        # partition train set into [train, val]
        if split == 'train':
            train, val = self.partition_train_set(self.imgs, num_imgs_per_class_val_split)
            if original_split == 'train':
                self.imgs = train
            if original_split == 'val':
                self.imgs = val

        # limit the number of images in train or test set since the limit was already applied to the val set
        if split in ['train', 'test']:
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
            ok_classes = shuffle(list(range(num_classes)), random_state=1234)
            ok_classes = ok_classes[:num_classes]
            ok_classes = set(ok_classes)

            clean_imgs = []
            for img_name, idx in self.imgs:
                if idx in ok_classes:
                    clean_imgs.append((img_name, idx))

            self.imgs = clean_imgs

        # shuffle again for final exit
        self.imgs = shuffle(self.imgs, random_state=1234)

        # list of class_nbs for each image
        idcs = [idx for _, idx in self.imgs]

        self.wnids = self.classes
        self.wnid_to_idx = {wnid: idx for idx, wnid in zip(idcs, self.wnids)}
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for clss, idx in zip(self.classes, idcs)
                             for cls in clss}

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
        if 'ILSVRC2012_devkit_t12.tar.gz' not in files:
            raise FileNotFoundError('devkit_path must point to the devkit file'
                                    'ILSVRC2012_devkit_t12.tar.gz. Download from here:'
                                    'http://www.image-net.org/challenges/LSVRC/2012/downloads')

        parse_devkit_archive(devkit_dir)
        print(f'meta.bin generated at {devkit_dir}/meta.bin')


def parse_devkit_archive(root, file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.
    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir():
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
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


def _verify_archive(root, file, md5):
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

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


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()
