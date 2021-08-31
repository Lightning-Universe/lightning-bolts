import math
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised.ssl_finetuner import RelicDALearner
# from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)



def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLRFinetuneTransform, SimCLRTrainDataTransform


    seed_everything(1234)
    parser = ArgumentParser()

    parser = SimCLR.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    wandb_logger = WandbLogger(project='simclr-finetune-cifar10', name='without data_augmentation')

    if args.dataset == 'cifar10':
        val_split = 5000
        # if args.num_nodes * args.gpus * args.batch_size > val_split:
        #     val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        )
        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        # args.gaussian_blur = True  # test relic.
        args.jitter_strength = 0.5

    args.use_relic_loss = True
    print('args.use_relic_loss: ', args.use_relic_loss)
    dm.train_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=False,
            use_relic_loss=args.use_relic_loss,
        )
    dm.val_transforms = SimCLRFinetuneTransform(
            normalize=cifar10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=True,
            use_relic_loss=args.use_relic_loss,
        )
    # dm.test_transforms = SimCLRFinetuneTransform(
    #     input_height=args.input_height,
    #     jitter_strength=args.jitter_strength,
    #     normalize=normalization,
    #     # use_relic_loss=args.use_relic_loss,
    # )

    backbone = SimCLR(
        gpus=args.gpus,
        nodes=1,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        maxpool1=args.maxpool1,
        first_conv=args.first_conv,
        dataset=args.dataset,
        use_relic_loss=args.use_relic_loss,
        ).load_from_checkpoint(args.ckpt_path, strict=False)

    # import ipdb; ipdb.set_trace()
    # add relic loss and data augmentation based on pre-trained SimCLR.
    model = RelicDALearner(
        backbone,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.online_ft = True
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=False,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    cli_main()
