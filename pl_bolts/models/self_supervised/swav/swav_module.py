"""Adapted from official swav implementation: https://github.com/facebookresearch/swav."""
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn

from pl_bolts.models.self_supervised.swav.loss import SWAVLoss
from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)


class SwAV(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3000,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        queue_length: int = 0,  # must be divisible by total batch-size
        queue_path: str = "queue",
        epoch_queue_starts: int = 15,
        crops_for_assign: tuple = (0, 1),
        nmb_crops: tuple = (2, 6),
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes: number of nodes to train on
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optimizer: optimizer to use
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.queue_length = queue_length
        self.queue_path = queue_path
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.model = self.init_model()
        self.criterion = SWAVLoss(
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            temperature=self.temperature,
            crops_for_assign=self.crops_for_assign,
            nmb_crops=self.nmb_crops,
            sinkhorn_iterations=self.sinkhorn_iterations,
            epsilon=self.epsilon,
        )
        self.use_the_queue = None
        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size
        self.queue = None

    def setup(self, stage):
        if self.queue_length > 0:
            queue_folder = os.path.join(self.logger.log_dir, self.queue_path)
            if not os.path.exists(queue_folder):
                os.makedirs(queue_folder)

            self.queue_path = os.path.join(queue_folder, "queue" + str(self.trainer.global_rank) + ".pth")

            if os.path.isfile(self.queue_path):
                self.queue = torch.load(self.queue_path)["queue"]

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
        )

    def forward(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

            if self.queue is not None:
                self.queue = self.queue.to(self.device)

        self.use_the_queue = False

    def on_train_epoch_end(self) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        inputs, y = batch
        inputs = inputs[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # SWAV loss computation
        loss, queue, use_queue = self.criterion(
            output=output,
            embedding=embedding,
            prototype_weights=self.model.prototypes.weight,
            batch_size=bs,
            queue=self.queue,
            use_queue=self.use_the_queue,
        )
        self.queue = queue
        self.use_the_queue = use_queue
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
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

        return [{"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.0}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

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
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="stl10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")
        parser.add_argument("--queue_path", type=str, default="queue", help="path for queue")

        parser.add_argument(
            "--nmb_crops", type=int, default=[2, 4], nargs="+", help="list of number of crops (example: [2, 6])"
        )
        parser.add_argument(
            "--size_crops", type=int, default=[96, 36], nargs="+", help="crops resolutions (example: [224, 96])"
        )
        parser.add_argument(
            "--min_scale_crops",
            type=float,
            default=[0.33, 0.10],
            nargs="+",
            help="argument in RandomResizedCrop (example: [0.14, 0.05])",
        )
        parser.add_argument(
            "--max_scale_crops",
            type=float,
            default=[1, 0.33],
            nargs="+",
            help="argument in RandomResizedCrop (example: [1., 0.14])",
        )

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        # swav params
        parser.add_argument(
            "--crops_for_assign",
            type=int,
            nargs="+",
            default=[0, 1],
            help="list of crops id used for computing assignments",
        )
        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument(
            "--epsilon", default=0.05, type=float, help="regularization parameter for Sinkhorn-Knopp algorithm"
        )
        parser.add_argument(
            "--sinkhorn_iterations", default=3, type=int, help="number of iterations in Sinkhorn-Knopp algorithm"
        )
        parser.add_argument("--nmb_prototypes", default=512, type=int, help="number of prototypes")
        parser.add_argument(
            "--queue_length",
            type=int,
            default=0,
            help="length of the queue (0 for no queue); must be divisible by total batch size",
        )
        parser.add_argument(
            "--epoch_queue_starts", type=int, default=15, help="from this epoch, we start using a queue"
        )
        parser.add_argument(
            "--freeze_prototypes_epochs",
            default=1,
            type=int,
            help="freeze the prototypes during this many epochs from the start",
        )

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform

    parser = ArgumentParser()

    # model args
    parser = SwAV.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False

        normalization = stl10_normalization()
    elif args.dataset == "cifar10":
        args.batch_size = 2
        args.num_workers = 0

        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False

        normalization = cifar10_normalization()

        # cifar10 specific params
        args.size_crops = [32, 16]
        args.nmb_crops = [2, 1]
        args.gaussian_blur = False
    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1.0, 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "lars"
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3

        args.nmb_prototypes = 3000
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.dims[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
    )

    # swav model init
    model = SwAV(**args.__dict__)

    online_evaluator = None
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
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
