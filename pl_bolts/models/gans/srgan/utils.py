from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data.dataset import random_split

from pl_bolts.datamodules.sr_datamodule import SRDataModule
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset
from pl_bolts.datasets.sr_celeba_dataset import SRCelebADataset
from pl_bolts.datasets.stl10_sr_dataset import SRSTL10Dataset


def parse_args(args, pl_module_cls):
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["celeba", "mnist", "stl10"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--save_model_checkpoint", dest="save_model_checkpoint", action="store_true")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "celeba":
        hr_image_size = 128
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 3
        dataset_cls = SRCelebADataset
        dataset_train = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="train"
        )
        dataset_val = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="valid"
        )
        dataset_test = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="test"
        )
    elif script_args.dataset == "mnist":
        hr_image_size = 28
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 1
        dataset_cls = SRMNISTDataset
        dataset_dev = dataset_cls(hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, train=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, train=False)
    elif script_args.dataset == "stl10":
        hr_image_size = 96
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 3
        dataset_cls = SRSTL10Dataset
        dataset_dev = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="train"
        )
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="test"
        )

    parser = SRDataModule.add_argparse_args(parser)
    parser = pl_module_cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    return args, image_channels, (dataset_train, dataset_val, dataset_test)
