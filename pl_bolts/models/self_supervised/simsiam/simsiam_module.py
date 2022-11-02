from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor

from pl_bolts.models.self_supervised.byol.models import MLP, SiameseArm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class SimSiam(LightningModule):
    """PyTorch Lightning implementation of Exploring Simple Siamese Representation Learning (SimSiam_)_

    Paper authors: Xinlei Chen, Kaiming He.

    Args:
        learning_rate (float, optional): optimizer leaning rate. Defaults to 0.05.
        weight_decay (float, optional): optimizer weight decay. Defaults to 1e-4.
        momentum (float, optional): optimizer momentum. Defaults to 0.9.
        warmup_epochs (int, optional): number of epochs for scheduler warmup. Defaults to 10.
        max_epochs (int, optional): maximum number of epochs for scheduler. Defaults to 100.
        base_encoder (Union[str, nn.Module], optional): base encoder architecture. Defaults to "resnet50".
        encoder_out_dim (int, optional): base encoder output dimension. Defaults to 2048.
        projector_hidden_dim (int, optional): projector MLP hidden dimension. Defaults to 2048.
        projector_out_dim (int, optional): project MLP output dimension. Defaults to 2048.
        predictor_hidden_dim (int, optional): predictor MLP hidden dimension. Defaults to 512.
        exclude_bn_bias (bool, optional): option to exclude batchnorm and bias terms from weight decay.
            Defaults to False.

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
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _SimSiam: https://arxiv.org/pdf/2011.10566v1.pdf
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        base_encoder: Union[str, nn.Module] = "resnet50",
        encoder_out_dim: int = 2048,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        exclude_bn_bias: bool = False,
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
        imgs, _ = batch
        img1, img2 = imgs[:2]

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

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_weight_decay(self.named_parameters(), weight_decay=self.hparams.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [scheduler]

    @staticmethod
    def exclude_from_weight_decay(named_params, weight_decay, skip_list=("bias", "bn")) -> List[Dict]:
        """Exclude parameters from weight decay."""
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif param.ndim == 1 or name in skip_list:
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        args = parser.parse_args([])

        if "max_epochs" in args:
            parser.set_defaults(max_epochs=100)
        else:
            parser.add_argument("--max_epochs", type=int, default=100)

        parser.add_argument("--learning_rate", default=0.05, type=float, help="base learning rate")
        parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
        parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
        parser.add_argument("--base_encoder", default="resnet50", type=str, help="encoder backbone")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

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
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet2012", "stl10"])

    args = parser.parse_args()

    # Initialize datamodule
    if args.dataset == "cifar10":
        dm = CIFAR10DataModule.from_argparse_args(args)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        args.num_classes = dm.num_classes
    elif args.dataset == "stl10":
        dm = STL10DataModule.from_argparse_args(args)
        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed

        (c, h, w) = dm.dims
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes
    elif args.dataset == "imagenet2012":
        dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = dm.dims
        dm.train_transforms = SimCLRTrainDataTransform(h)
        dm.val_transforms = SimCLREvalDataTransform(h)
        args.num_classes = dm.num_classes
    else:
        raise ValueError(
            f"{args.dataset} is not a valid dataset. Dataset must be 'cifar10', 'stl10', or 'imagenet2012'."
        )

    # Initialize SimSiam module
    model = SimSiam(**vars(args))

    # Finetune in real-time
    online_eval = SSLOnlineEvaluator(dataset=args.dataset, z_dim=2048, num_classes=dm.num_classes)

    trainer = Trainer.from_argparse_args(args, callbacks=[online_eval])

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
