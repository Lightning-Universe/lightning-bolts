import os
from argparse import ArgumentParser

from lightning import Trainer, seed_everything

from pl_bolts.models.self_supervised import CPC_v2, SSLFineTuner
from pl_bolts.transforms.self_supervised.cpc_transforms import (
    CPCEvalTransformsCIFAR10,
    CPCEvalTransformsSTL10,
    CPCTrainTransformsCIFAR10,
    CPCTrainTransformsSTL10,
)
from pl_bolts.utils.stability import under_review


@under_review()
def cli_main():  # pragma: no cover
    from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule

    seed_everything(1234)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--dataset", type=str, help="stl10, cifar10", default="cifar10")
    parser.add_argument("--ckpt_path", type=str, help="path to ckpt")
    parser.add_argument("--data_dir", type=str, help="path to ckpt", default=os.getcwd())
    args = parser.parse_args()

    # load the backbone
    backbone = CPC_v2.load_from_checkpoint(args.ckpt_path, strict=False)

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = CPCTrainTransformsCIFAR10()
        dm.val_transforms = CPCEvalTransformsCIFAR10()
        dm.test_transforms = CPCEvalTransformsCIFAR10()

    elif args.dataset == "stl10":
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        dm.train_transforms = CPCTrainTransformsSTL10()
        dm.val_transforms = CPCEvalTransformsSTL10()
        dm.test_transforms = CPCEvalTransformsSTL10()

    # finetune
    tuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)
    trainer = Trainer.from_argparse_args(args, early_stop_callback=True)
    trainer.fit(tuner, dm)

    trainer.test(datamodule=dm)


if __name__ == "__main__":
    cli_main()
