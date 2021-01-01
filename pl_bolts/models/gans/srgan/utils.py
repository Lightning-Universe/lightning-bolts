from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence, Text, Tuple

import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torchvision.datasets.vision import VisionDataset

from pl_bolts.datamodules import SRDataModule
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset
from pl_bolts.datasets.sr_celeba_dataset import SRCelebADataset
from pl_bolts.datasets.sr_stl10_dataset import SRSTL10Dataset


def parse_args(
    args: Optional[Sequence[Text]], pl_module_cls: pl.LightningModule
) -> Tuple[Namespace, int, List[VisionDataset, VisionDataset, VisionDataset]]:

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["celeba", "mnist", "stl10"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--save_model_checkpoint", dest="save_model_checkpoint", action="store_true")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "celeba":
        dataset_cls = SRCelebADataset
        dataset_train = dataset_cls(script_args.scale_factor, root=script_args.data_dir, split="train")
        dataset_val = dataset_cls(script_args.scale_factor, root=script_args.data_dir, split="valid")
        dataset_test = dataset_cls(script_args.scale_factor, root=script_args.data_dir, split="test")

    elif script_args.dataset == "mnist":
        dataset_cls = SRMNISTDataset
        dataset_dev = dataset_cls(script_args.scale_factor, root=script_args.data_dir, train=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(script_args.scale_factor, root=script_args.data_dir, train=False)

    elif script_args.dataset == "stl10":
        dataset_cls = SRSTL10Dataset
        dataset_dev = dataset_cls(script_args.scale_factor, root=script_args.data_dir, split="train")
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(script_args.scale_factor, root=script_args.data_dir, split="test")

    parser = SRDataModule.add_argparse_args(parser)
    parser = pl_module_cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    return args, (dataset_train, dataset_val, dataset_test)
