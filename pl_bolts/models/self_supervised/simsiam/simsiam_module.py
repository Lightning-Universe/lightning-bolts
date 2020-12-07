from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim import Adam

from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)


class SimSiam(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Exploring Simple Siamese Representation Learning (SimSiam)
    <https://arxiv.org/pdf/2011.10566v1.pdf>`_

    Paper authors: Xinlei Chen, Kaiming He.

    Model implemented by:
        - `Zvi Lapp <https://github.com/zlapp>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet

    Example::

        model = SimSiam()

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = pl.Trainer()
        trainer.fit(model, dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

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
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()

        self.online_network = SiameseArm()
        self._init_target_network()

    def _init_target_network(self):
        self.target_network = deepcopy(self.online_network)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._init_target_network()

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1).mean()
        return sim

    def shared_step(self, batch, batch_idx):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.target_network(img_2)
        loss_a = -1.0 * self.cosine_similarity(h1, z2)

        # Image 2 to image 1 loss
        _, z1, h1 = self.online_network(img_2)
        _, z2, h2 = self.target_network(img_1)
        loss_b = -1.0 * self.cosine_similarity(h1, z2)

        # Final loss
        total_loss = loss_a / 2.0 + loss_b / 2.0

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({"1_2_loss": loss_a, "2_1_loss": loss_b, "train_loss": total_loss})

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({"1_2_loss": loss_a, "2_1_loss": loss_b, "train_loss": total_loss})

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.online_network.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument(
            "--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head"
        )
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--fast_dev_run", action="store_true")
        parser.add_argument("--nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument(
            "--lars_wrapper", action="store_true", help="apple lars wrapper over optimizer used"
        )
        parser.add_argument(
            "--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay"
        )
        parser.add_argument(
            "--max_epochs", default=100, type=int, help="number of total epochs to run"
        )
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument(
            "--temperature", default=0.1, type=float, help="temperature parameter in training loss"
        )
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument(
            "--start_lr", default=0, type=float, help="initial warmup learning rate"
        )
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import (
        CIFAR10DataModule,
        ImagenetDataModule,
        STL10DataModule,
    )
    from pl_bolts.models.self_supervised.simclr import (
        SimCLREvalDataTransform,
        SimCLRTrainDataTransform,
    )

    seed_everything(1234)

    parser = ArgumentParser()

    # model args
    parser = SimSiam.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init datamodule
    if args.dataset == "stl10":
        dm = STL10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 5000
        if args.nodes * args.gpus * args.batch_size > val_split:
            val_split = args.nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "sgd"
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimSiam(**args.__dict__)

    # finetune in real-time
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

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.nodes,
        distributed_backend="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=[online_evaluator] if args.online_ft else None,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    cli_main()
