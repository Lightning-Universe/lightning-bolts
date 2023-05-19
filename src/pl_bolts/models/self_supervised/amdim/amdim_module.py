import os
from argparse import ArgumentParser
from typing import Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import optim
from torch.utils.data import DataLoader

from pl_bolts.losses.self_supervised_learning import FeatureMapContrastiveTask
from pl_bolts.models.self_supervised.amdim.datasets import AMDIMPretraining
from pl_bolts.models.self_supervised.amdim.networks import AMDIMEncoder
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pl_bolts.utils.stability import under_review


@under_review()
def generate_power_seq(lr, nb):
    half = int(nb / 2)
    coefs = [2**pow for pow in range(half, -half - 1, -1)]
    lrs = [lr * coef for coef in coefs]
    return lrs


# CIFAR 10
LEARNING_RATE_CIFAR = 2e-4
DATASET_CIFAR10 = {
    "dataset": "cifar10",
    "ndf": 320,
    "n_rkhs": 1280,
    "depth": 10,
    "image_height": 32,
    "batch_size": 200,
    "nb_classes": 10,
    "lr_options": generate_power_seq(LEARNING_RATE_CIFAR, 11),
}

# stl-10
LEARNING_RATE_STL = 2e-4
DATASET_STL10 = {
    "dataset": "stl10",
    "ndf": 192,
    "n_rkhs": 1536,
    "depth": 8,
    "image_height": 64,
    "batch_size": 200,
    "nb_classes": 10,
    "lr_options": generate_power_seq(LEARNING_RATE_STL, 11),
}

LEARNING_RATE_IMAGENET = 2e-4
DATASET_IMAGENET2012 = {
    "dataset": "imagenet2012",
    "ndf": 320,
    "n_rkhs": 2560,
    "depth": 10,
    "image_height": 128,
    "batch_size": 200,
    "nb_classes": 1000,
    "lr_options": generate_power_seq(LEARNING_RATE_IMAGENET, 11),
}


@under_review()
class AMDIM(LightningModule):
    """PyTorch Lightning implementation of Augmented Multiscale Deep InfoMax (AMDIM_)

    Paper authors: Philip Bachman, R Devon Hjelm, William Buchwalter.

    Model implemented by: `William Falcon <https://github.com/williamFalcon>`_

    This code is adapted to Lightning using the original author repo
    (`the original repo <https://github.com/Philip-Bachman/amdim-public>`_).

    Example:

        >>> from pl_bolts.models.self_supervised import AMDIM
        ...
        >>> model = AMDIM(encoder='resnet18')

    Train::

        trainer = Trainer()
        trainer.fit(model)

    .. _AMDIM: https://arxiv.org/abs/1906.00910
    """

    def __init__(
        self,
        datamodule: Union[str, LightningDataModule] = "cifar10",
        encoder: Union[str, torch.nn.Module, LightningModule] = "amdim_encoder",
        contrastive_task: Union[FeatureMapContrastiveTask] = FeatureMapContrastiveTask("01, 02, 11"),
        image_channels: int = 3,
        image_height: int = 32,
        encoder_feature_dim: int = 320,
        embedding_fx_dim: int = 1280,
        conv_block_depth: int = 10,
        use_bn: bool = False,
        tclip: int = 20.0,
        learning_rate: int = 2e-4,
        data_dir: str = "",
        num_classes: int = 10,
        batch_size: int = 200,
        num_workers: int = 16,
        **kwargs,
    ):
        """
        Args:
            datamodule: A LightningDatamodule
            encoder: an encoder string or model
            image_channels: 3
            image_height: pixels
            encoder_feature_dim: Called `ndf` in the paper, this is the representation size for the encoder.
            embedding_fx_dim: Output dim of the embedding function (`nrkhs` in the paper)
                (Reproducing Kernel Hilbert Spaces).
            conv_block_depth: Depth of each encoder block,
            use_bn: If true will use batchnorm.
            tclip: soft clipping non-linearity to the scores after computing the regularization term
                and before computing the log-softmax. This is the 'second trick' used in the paper
            learning_rate: The learning rate
            data_dir: Where to store data
            num_classes: How many classes in the dataset
            batch_size: The batch size
        """
        super().__init__()
        self.save_hyperparameters()

        # init encoder
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder = self.init_encoder()

        # the task
        self.contrastive_task = contrastive_task

        self.tng_split = None
        self.val_split = None

    def init_encoder(self):
        dummy_batch = torch.zeros(
            (2, self.hparams.image_channels, self.hparams.image_height, self.hparams.image_height)
        )
        encoder_name = self.hparams.encoder

        if encoder_name == "amdim_encoder":
            encoder = AMDIMEncoder(
                dummy_batch,
                num_channels=self.hparams.image_channels,
                encoder_feature_dim=self.hparams.encoder_feature_dim,
                embedding_fx_dim=self.hparams.embedding_fx_dim,
                conv_block_depth=self.hparams.conv_block_depth,
                encoder_size=self.hparams.image_height,
                use_bn=self.hparams.use_bn,
            )
            encoder.init_weights()
            return encoder
        return torchvision_ssl_encoder(encoder_name, return_all_feature_maps=True)

    def forward(self, img_1, img_2):
        # feats for img 1
        # r1 = last layer out
        # r5 = last layer with (b, c, 5, 5) size
        # r7 = last layer with (b, c, 7, 7) size
        maps = self.encoder(img_1)
        if len(maps) > 3:
            maps = maps[-3:]
        r1_x1, r5_x1, r7_x1 = maps

        # feats for img 2
        maps = self.encoder(img_2)
        if len(maps) > 3:
            maps = maps[-3:]
        r1_x2, r5_x2, r7_x2 = maps

        # first number = resnet block. second = image 1 or 2
        return r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2

    def training_step(self, batch, batch_nb):
        [img_1, img_2], _ = batch

        # ------------------
        # FEATURE EXTRACTION
        # extract features from various blocks for each image
        # _x1 are from image 1
        # _x2 from image 2
        r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2 = self.forward(img_1, img_2)

        result = {
            "r1_x1": r1_x1,
            "r5_x1": r5_x1,
            "r7_x1": r7_x1,
            "r1_x2": r1_x2,
            "r5_x2": r5_x2,
            "r7_x2": r7_x2,
        }

        return result

    def training_step_end(self, outputs):
        r1_x1 = outputs["r1_x1"]
        r5_x1 = outputs["r5_x1"]
        r7_x1 = outputs["r7_x1"]
        r1_x2 = outputs["r1_x2"]
        r5_x2 = outputs["r5_x2"]
        r7_x2 = outputs["r7_x2"]

        # Contrastive task
        loss, lgt_reg = self.contrastive_task((r1_x1, r5_x1, r7_x1), (r1_x2, r5_x2, r7_x2))
        unsupervised_loss = loss.sum() + lgt_reg

        # ------------------
        # FULL LOSS
        total_loss = unsupervised_loss

        tensorboard_logs = {"train_nce_loss": total_loss}
        result = {"loss": total_loss, "log": tensorboard_logs}

        return result

    def validation_step(self, batch, batch_nb):
        [img_1, img_2], labels = batch

        # generate features
        r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2 = self.forward(img_1, img_2)

        # Contrastive task
        loss, lgt_reg = self.contrastive_task((r1_x1, r5_x1, r7_x1), (r1_x2, r5_x2, r7_x2))
        unsupervised_loss = loss.sum() + lgt_reg

        result = {"val_nce": unsupervised_loss}
        return result

    def validation_epoch_end(self, outputs):
        val_nce = 0
        for output in outputs:
            val_nce += output["val_nce"]

        val_nce = val_nce / len(outputs)
        tensorboard_logs = {"val_nce": val_nce}
        return {"val_loss": val_nce, "log": tensorboard_logs}

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(), lr=self.hparams.learning_rate, betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-7
        )

        # if self.hparams.datamodule in ['cifar10', 'stl10', 'cifar100']:
        #     lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        # else:
        #     lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return opt  # [opt], [lr_scheduler]

    def train_dataloader(self):
        kwargs = dict(nb_classes=self.hparams.nb_classes) if self.hparams.datamodule == "imagenet2012" else {}
        dataset = AMDIMPretraining.get_dataset(self.hparams.datamodule, self.hparams.data_dir, split="train", **kwargs)

        # LOADER
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )
        return loader

    def val_dataloader(self):
        kwargs = dict(nb_classes=self.hparams.nb_classes) if self.hparams.datamodule == "imagenet2012" else {}
        dataset = AMDIMPretraining.get_dataset(self.hparams.datamodule, self.hparams.data_dir, split="val", **kwargs)

        # LOADER
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--datamodule", type=str, default="cifar10")

        DATASETS = {"cifar10": DATASET_CIFAR10, "stl10": DATASET_STL10, "imagenet2012": DATASET_IMAGENET2012}

        (args, _) = parser.parse_known_args()
        dataset = DATASETS[args.datamodule]

        # dataset options
        parser.add_argument("--num_classes", default=dataset["nb_classes"], type=int)

        # network params
        parser.add_argument("--tclip", type=float, default=20.0, help="soft clipping range for NCE scores")
        parser.add_argument("--use_bn", type=int, default=0)
        parser.add_argument("--encoder_feature_dim", type=int, default=dataset["ndf"], help="feature size for encoder")
        parser.add_argument(
            "--embedding_fx_dim",
            type=int,
            default=dataset["n_rkhs"],
            help="number of dimensions in fake RKHS embeddings",
        )
        parser.add_argument("--conv_block_depth", type=int, default=dataset["depth"])
        parser.add_argument("--image_height", type=int, default=dataset["image_height"])

        # trainin params
        # resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        #            'resnext50_32x4d', 'resnext101_32x8d',
        #            'wide_resnet50_2', 'wide_resnet101_2']
        parser.add_argument(
            "--batch_size", type=int, default=dataset["batch_size"], help="input batch size (default: 200)"
        )
        parser.add_argument("--learning_rate", type=float, default=0.0002)

        # data
        parser.add_argument("--data_dir", default=os.getcwd(), type=str)
        parser.add_argument("--num_workers", type=int, default=16)
        return parser


@under_review()
def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = AMDIM.add_model_specific_args(parser)

    args = parser.parse_args()

    model = AMDIM(**vars(args), encoder="resnet18")
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
