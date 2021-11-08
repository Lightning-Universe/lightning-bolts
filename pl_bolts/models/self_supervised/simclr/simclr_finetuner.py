import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)


def cli_main():  # pragma: no cover
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

    seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="cifar10, stl10, imagenet", default="cifar10")
    parser.add_argument("--ckpt_path", type=str, help="path to ckpt")
    parser.add_argument("--data_dir", type=str, help="path to dataset", default=os.getcwd())

    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=4, type=int, help="number of GPUs")
    parser.add_argument("--num_epochs", default=100, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument("--in_features", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--nesterov", type=bool, default=False)  # fix nesterov flag here
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--final_lr", type=float, default=0.0)

    args = parser.parse_args()

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(), input_height=dm.size()[-1], eval_transform=False
        )
        dm.val_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        dm.test_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )

        args.maxpool1 = False
        args.first_conv = False
        args.num_samples = 1
    elif args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        args.num_samples = 1

        dm.train_transforms = SimCLRFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=False
        )
        dm.val_transforms = SimCLRFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        dm.test_transforms = SimCLRFinetuneTransform(
            normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
        )

        args.maxpool1 = False
        args.first_conv = True
    elif args.dataset == "imagenet":
        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_transforms = SimCLRFinetuneTransform(
            normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=False
        )
        dm.val_transforms = SimCLRFinetuneTransform(
            normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
        )
        dm.test_transforms = SimCLRFinetuneTransform(
            normalize=imagenet_normalization(), input_height=dm.size()[-1], eval_transform=True
        )

        args.num_samples = 1
        args.maxpool1 = True
        args.first_conv = True
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        maxpool1=args.maxpool1,
        first_conv=args.first_conv,
        dataset=args.dataset,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = SSLFineTuner(
        backbone,
        in_features=args.in_features,
        num_classes=dm.num_classes,
        epochs=args.num_epochs,
        hidden_dim=None,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        final_lr=args.final_lr,
    )

    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=1,
        precision=16,
        max_epochs=args.num_epochs,
        accelerator="ddp",
        sync_batchnorm=True if args.gpus > 1 else False,
    )

    trainer.fit(tuner, dm)
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    cli_main()
