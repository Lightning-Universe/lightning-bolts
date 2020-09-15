import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from pl_bolts.models.self_supervised import SimCLR, SSLFineTuner
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


def cli_main():  # pragma: no-cover
    from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--dataset', type=str, help='stl10, cifar10', default='cifar10')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
    parser.add_argument('--data_dir', type=str, help='path to ckpt', default=os.getcwd())
    args = parser.parse_args()

    # load the backbone
    backbone = SimCLR.load_from_checkpoint(args.ckpt_path, strict=False)

    # init default datamodule
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        dm.test_transforms = SimCLREvalDataTransform(32)
        args.num_samples = dm.num_samples

    elif args.dataset == 'stl10':
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        args.num_samples = dm.num_labeled_samples

        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)

    elif args.dataset == 'imagenet2012':
        dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = dm.size()
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)

    # finetune
    tuner = SSLFineTuner(backbone, in_features=2048 * 2 * 2, num_classes=dm.num_classes, hidden_dim=None)
    trainer = pl.Trainer.from_argparse_args(args, early_stop_callback=True)
    trainer.fit(tuner, dm)

    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
