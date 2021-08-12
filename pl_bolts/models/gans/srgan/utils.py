from torch.utils.data.dataset import random_split

from pl_bolts.datasets.sr_celeba_dataset import SRCelebA
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST
from pl_bolts.datasets.sr_stl10_dataset import SRSTL10


def prepare_datasets(dataset, scale_factor, data_dir):
    assert dataset in ["celeba", "mnist", "stl10"]

    if dataset == "celeba":
        dataset_cls = SRCelebA
        dataset_train = dataset_cls(scale_factor, root=data_dir, split="train", download=True)
        dataset_val = dataset_cls(scale_factor, root=data_dir, split="valid", download=True)
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test", download=True)

    elif dataset == "mnist":
        dataset_cls = SRMNIST
        dataset_dev = dataset_cls(scale_factor, root=data_dir, train=True, download=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(scale_factor, root=data_dir, train=False, download=True)

    elif dataset == "stl10":
        dataset_cls = SRSTL10
        dataset_dev = dataset_cls(scale_factor, root=data_dir, split="train", download=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test", download=True)

    return (dataset_train, dataset_val, dataset_test)
