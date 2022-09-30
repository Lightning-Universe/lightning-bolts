from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor

from pl_bolts.models.self_supervised.byol import MLP, SiameseArm
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class SimSiam(LightningModule):
    """PyTorch Lightning implementation of Exploring Simple Siamese Representation Learning (SimSiam_)_

    Paper authors: Xinlei Chen, Kaiming He.

    Args:

    Model implemented by:
        - `Zvi Lapp <https://github.com/zlapp>`_

    Example::

        model = SimSiam()

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = Trainer()
        trainer.fit(model, datamodule=dm)

    CLI command::

        # cifar10
        python simsiam_module.py --gpus 1

        # imagenet
        python simsiam_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf
    """

    def __init__(
        self,
        base_learning_rate: float = 0.05,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        base_encoder: Union[str, nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        **kwargs,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore="base_encoder")

        self.online_network = SiameseArm(base_encoder, encoder_out_dim, projector_hidden_dim, projector_out_dim)
        self.target_network = deepcopy(self.online_network)
        self.predictor = MLP(projector_out_dim, predictor_hidden_dim, projector_out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Returns encoded representation of a view."""
        return self.online_network.encode(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete training loop."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete validation loop."""
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        """Shared evaluation step for training and validation loops."""
        (img1, img2), _ = batch

        # Calculate similarity loss in each direction
        loss_12 = self.calculate_loss(img1, img2)
        loss_21 = self.calculate_loss(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        # Log loss
        if step == "train":
            self.log_dict({"train_loss_12": loss_12, "train_loss_21": loss_21, "train_loss": total_loss})
        elif step == "val":
            self.log_dict({"val_loss_12": loss_12, "val_loss_21": loss_21, "val_loss": total_loss})
        else:
            raise ValueError(f"Step '{step}' is invalid. Must be 'train' or 'val'.")

        return total_loss

    def calculate_loss(self, v_online: Tensor, v_target: Tensor) -> Tensor:
        """Calculates similarity loss between the online network prediction of target network projection.

        Args:
            v_online (Tensor): Online network view
            v_target (Tensor): Target network view
        """
        _, z1 = self.online_network(v_online)
        h1 = self.predictor(z1)
        with torch.no_grad():
            _, z2 = self.target_network(v_target)
        loss = -0.5 * F.cosine_similarity(h1, z2).mean()

        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.hparams.learning_rate, momentum=0.9, weight_decay=self.hparams.weight_decay
            )
        elif self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Optimizer {self.optim} is not supported.")

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--base_encoder", default="resnet50", type=str, help="encoder backbone")
        parser.add_argument("--base_learning_rate", default=0.05, type=float, help="base learning rate")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of max epochs")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    seed_everything(1234)

    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)
    parser = SimSiam.add_model_specific_args(parser)
    parser = CIFAR10DataModule.add_dataset_specific_args(parser)

    args = parser.parse_args()

    # Initialize datamodule
    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.dims[-1]

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.dims[-1]
        args.temperature = 0.5

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "lars"
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.dims[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
    )

    model = SimSiam(**vars(args))

    # Finetune in real-time
    online_evaluator = None
    if args.online_ft:
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

    trainer = Trainer.from_argparse_args(
        args,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
