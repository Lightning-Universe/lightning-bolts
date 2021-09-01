import math
from argparse import ArgumentParser
import wandb 
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.nn import functional as F

# from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.models.self_supervised.ssl_finetuner import RelicDALearner
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

import os
from pytorch_lightning import Trainer, seed_everything
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner



def cli_main():
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

    seed_everything(1234)

    parser = ArgumentParser()

    # wandb params
    parser.add_argument("--project", type=str, help="wandb project name", default="simclr-cifar10")
    parser.add_argument("--name", type=str, help="wandb run name.", default="testing")
    # relic params
    parser.add_argument("--use_relic_loss", type=bool, help="to use_relic_loss.", default=False)
    parser.add_argument("--alfa", type=float, help="how depend on relic loss.", default=0.1)

    parser.add_argument("--dataset", type=str, help="cifar10, stl10, imagenet", default="cifar10")
    parser.add_argument("--ckpt_path", type=str, help="path to ckpt")
    parser.add_argument("--data_dir", type=str, help="path to dataset", default=os.getcwd())

    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=4, type=int, help="number of GPUs")
    parser.add_argument("--num_epochs", default=200, type=int, help="number of epochs")

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
    wandb.init(project=args.project, name=args.name)
    wandb_logger = WandbLogger(project=args.project, name=args.name)

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=False,
            use_relic_loss=args.use_relic_loss,
        )
        dm.val_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=False,
            use_relic_loss=args.use_relic_loss,
        )

        args.maxpool1 = False
        args.first_conv = False
        args.num_samples = 1

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        maxpool1=args.maxpool1,
        first_conv=args.first_conv,
        dataset=args.dataset,
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    # add relic loss and data augmentation based on pre-trained SimCLR.
    model = RelicDALearner(
        backbone,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    #############################################################################

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss", filename="relic-{epoch:02d}-{val_loss:.2f}")
    callbacks = [model_checkpoint, lr_monitor]

    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=1,
        precision=16,
        max_epochs=args.num_epochs,
        distributed_backend="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
        fast_dev_run=False,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    cli_main()
