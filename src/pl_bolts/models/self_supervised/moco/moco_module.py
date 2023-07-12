"""Adapted from: https://github.com/facebookresearch/moco.

Original work is: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved This implementation is: Copyright
(c) PyTorch Lightning, Inc. and its affiliates. All Rights Reserved

This implementation is licensed under Attribution-NonCommercial 4.0 International; You may not use this file except in
compliance with the License.

You may obtain a copy of the License from the LICENSE file present in this folder.
"""
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# It seems to be impossible to avoid mypy errors if using import instead of getattr().
# See https://github.com/python/mypy/issues/8823
try:
    LRScheduler: Any = getattr(optim.lr_scheduler, "LRScheduler")
except AttributeError:
    LRScheduler = getattr(optim.lr_scheduler, "_LRScheduler")

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.metrics import precision_at_k
from pl_bolts.models.self_supervised.moco.utils import concatenate_all, shuffle_batch, sort_batch, validate_batch
from pl_bolts.transforms.self_supervised.moco_transforms import (
    MoCo2EvalCIFAR10Transforms,
    MoCo2TrainCIFAR10Transforms,
)
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class RepresentationQueue(nn.Module):
    """The queue is implemented as list of representations and a pointer to the location where the next batch of
    representations will be overwritten."""

    def __init__(self, representation_size: int, queue_size: int):
        super().__init__()

        self.representations: Tensor
        self.register_buffer("representations", torch.randn(representation_size, queue_size))
        self.representations = nn.functional.normalize(self.representations, dim=0)

        self.pointer: Tensor
        self.register_buffer("pointer", torch.zeros([], dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, x: Tensor) -> None:
        """Replaces representations in the queue, starting at the current queue pointer, and advances the pointer.

        Args:
            x: A mini-batch of representations. The queue size has to be a multiple of the total number of
                representations across all devices.

        """
        # Gather representations from all GPUs into a [batch_size * world_size, num_features] tensor, in case of
        # distributed training.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            x = concatenate_all(x)

        queue_size = self.representations.shape[1]
        batch_size = x.shape[0]
        if queue_size % batch_size != 0:
            raise ValueError(f"Queue size ({queue_size}) is not a multiple of the batch size ({batch_size}).")

        end = self.pointer + batch_size
        self.representations[:, int(self.pointer) : int(end)] = x.T
        self.pointer = end % queue_size


class MoCo(LightningModule):
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "resnet18",
        head: Optional[nn.Module] = None,
        representation_size: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        temperature: float = 0.07,
        exclude_bn_bias: bool = False,
        optimizer: Type[optim.Optimizer] = optim.SGD,
        optimizer_params: Optional[Dict[str, Any]] = None,
        lr_scheduler: Type[LRScheduler] = optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A module that trains an encoder using Momentum Contrast.

        *MoCo paper*: `Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick
        <https://arxiv.org/abs/1911.05722>`_

        *Moco v2 paper*: `Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He <https://arxiv.org/abs/2003.04297>`_

        *Adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by*:
        `William Falcon <https://github.com/williamFalcon>`_

        *Refactored by*: `Seppo Enarvi <https://github.com/senarvi>`_

        Example::
            from pl_bolts.models.self_supervised import MoCo
            model = MoCo()
            trainer = Trainer()
            trainer.fit(model)

        CLI command::
            python moco_module.py fit \
                --data.data_dir /path/to/imagenet \
                --data.batch_size 32 \
                --data.num_workers 4 \
                --trainer.accelerator gpu \
                --trainer.devices 8

        Args:
            encoder: The encoder module. Either a Torchvision model name or a ``torch.nn.Module``.
            head: An optional projection head that will be appended to the encoder during training.
            representation_size: Size of a feature vector produced by the projection head (or in case a projection head
                is not used, the encoder).
            num_negatives: Number of negative examples to be kept in the queue.
            encoder_momentum: Momentum for updating the key encoder.
            temperature: The temperature parameter for the MoCo loss.
            exclude_bn_bias: If ``True``, weight decay will be applied only to convolutional layer weights.
            optimizer: Which optimizer class to use for training.
            optimizer_params: Parameters to pass to the optimizer constructor.
            lr_scheduler: Which learning rate scheduler class to use for training.
            lr_scheduler_params: Parameters to pass to the learning rate scheduler constructor.

        """
        super().__init__()

        self.num_negatives = num_negatives
        self.encoder_momentum = encoder_momentum
        self.temperature = temperature
        self.exclude_bn_bias = exclude_bn_bias
        self.optimizer_class = optimizer
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
        else:
            self.optimizer_params = {"lr": 0.03, "momentum": 0.9, "weight_decay": 1e-4}
        self.lr_scheduler_class = lr_scheduler
        if lr_scheduler_params is not None:
            self.lr_scheduler_params = lr_scheduler_params
        else:
            self.lr_scheduler_params = {"T_max": 100}

        if isinstance(encoder, str):
            template_model = getattr(torchvision.models, encoder)
            self.encoder_q = template_model(num_classes=representation_size)
        else:
            self.encoder_q = encoder
        self.encoder_k = deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        if head is not None:
            self.head_q: Optional[nn.Module] = head
            self.head_k: Optional[nn.Module] = deepcopy(head)
            for param in self.head_k.parameters():
                param.requires_grad = False
        else:
            self.head_q = None
            self.head_k = None

        # Two different queues of representations are needed, one for training and one for validation data.
        self.queue = RepresentationQueue(representation_size, num_negatives)
        self.val_queue = RepresentationQueue(representation_size, num_negatives)

    def forward(self, query_images: Tensor, key_images: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the forward passes of both encoders and projection heads.

        Args:
            query_images: A mini-batch of query images in a ``[batch_size, num_channels, height, width]`` tensor.
            key_images: A mini-batch of key images in a ``[batch_size, num_channels, height, width]`` tensor.

        Returns:
            A tuple of query and key representations.

        """
        q = self.encoder_q(query_images)
        if self.head_q is not None:
            q = self.head_q(q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            # The keys are shuffled between the GPUs before encoding them, to avoid batch normalization leaking
            # information between the samples. This works only when using the DDP strategy.
            if isinstance(self.trainer.strategy, DDPStrategy):
                key_images, original_order = shuffle_batch(key_images)

            k = self.encoder_k(key_images)
            if self.head_k is not None:
                k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)

            if isinstance(self.trainer.strategy, DDPStrategy):
                k = sort_batch(k, original_order)

        return q, k

    def training_step(self, batch: Tuple[List[List[Tensor]], List[Any]], batch_idx: int) -> STEP_OUTPUT:
        images = validate_batch(batch)
        self._momentum_update_key_encoder()
        loss, acc1, acc5 = self._calculate_loss(images, self.queue)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/acc1", acc1, sync_dist=True)
        self.log("train/acc5", acc5, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[List[List[Tensor]], List[Any]], batch_idx: int) -> Optional[STEP_OUTPUT]:
        images = validate_batch(batch)
        loss, acc1, acc5 = self._calculate_loss(images, self.val_queue)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/acc1", acc1, sync_dist=True)
        self.log("val/acc5", acc5, sync_dist=True)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        """Constructs the optimizer and learning rate scheduler based on ``self.optimizer_params`` and
        ``self.lr_scheduler_params``.

        If weight decay is specified, it will be applied only to convolutional layer weights.
        """
        if (
            ("weight_decay" in self.optimizer_params)
            and (self.optimizer_params["weight_decay"] != 0)
            and self.exclude_bn_bias
        ):
            defaults = copy(self.optimizer_params)
            weight_decay = defaults.pop("weight_decay")

            wd_group = []
            nowd_group = []
            for name, tensor in self.named_parameters():
                if not tensor.requires_grad:
                    continue
                if ("bias" in name) or ("bn" in name):
                    nowd_group.append(tensor)
                else:
                    wd_group.append(tensor)

            params = [
                {"params": wd_group, "weight_decay": weight_decay},
                {"params": nowd_group, "weight_decay": 0.0},
            ]
            optimizer = self.optimizer_class(params, **defaults)
        else:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Momentum update of the key encoder."""
        momentum = self.encoder_momentum
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def _calculate_loss(self, images: Tensor, queue: RepresentationQueue) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the normalized temperature-scaled cross entropy loss from a mini-batch of image pairs.

        Args:
            images: A mini-batch of image pairs in a ``[batch_size, 2, num_channels, height, width]`` tensor.
            queue: The queue that the query representations will be compared against. The key representations will be
                added to the queue.

        """
        if images.size(1) != 2:
            raise ValueError(
                f"MoCo expects two transformations of every image. Got {images.size(1)} transformations of an image."
            )

        query_images = images[:, 0]
        key_images = images[:, 1]
        q, k = self(query_images, key_images)

        # Concatenate logits from the positive pairs (batch_size x 1) and the negative pairs (batch_size x queue_size).
        pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        neg_logits = torch.einsum("nc,ck->nk", [q, queue.representations.clone().detach()])
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits /= self.temperature

        # The correct label for every query is 0. Calculate the cross entropy of classifying each query correctly.
        target_idxs = torch.zeros(logits.shape[0], dtype=torch.long).type_as(logits)
        loss = F.cross_entropy(logits, target_idxs.long())
        acc1, acc5 = precision_at_k(logits, target_idxs, top_k=(1, 5))

        queue.dequeue_and_enqueue(k)
        return loss, acc1, acc5


def collate(samples: List[Tuple[Tuple[Tensor, Tensor], int]]) -> Tuple[List[Tuple[Tensor, Tensor]], List[int]]:
    return tuple(zip(*samples))  # type: ignore


class CIFAR10ContrastiveDataModule(CIFAR10DataModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            train_transforms=MoCo2TrainCIFAR10Transforms(),
            val_transforms=MoCo2EvalCIFAR10Transforms(),
            **kwargs,
        )

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate,
        )


def cli_main() -> None:
    from pytorch_lightning.cli import LightningCLI

    LightningCLI(MoCo, CIFAR10ContrastiveDataModule, seed_everything_default=42)


if __name__ == "__main__":
    cli_main()
