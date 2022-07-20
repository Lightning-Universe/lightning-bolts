"""
CPC V2
======
"""
import math
from argparse import ArgumentParser
from typing import Optional

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_warn
from torch import optim

from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.losses.self_supervised_learning import CPCTask
from pl_bolts.models.self_supervised.cpc.networks import cpc_resnet101
from pl_bolts.models.self_supervised.cpc.transforms import (
    CPCEvalTransformsCIFAR10,
    CPCEvalTransformsImageNet128,
    CPCEvalTransformsSTL10,
    CPCTrainTransformsCIFAR10,
    CPCTrainTransformsImageNet128,
    CPCTrainTransformsSTL10,
)
from pl_bolts.utils.pretrained_weights import load_pretrained
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pl_bolts.utils.stability import under_review

__all__ = ["CPC_v2"]


@under_review()
class CPC_v2(LightningModule):
    def __init__(
        self,
        encoder_name: str = "cpc_encoder",
        patch_size: int = 8,
        patch_overlap: int = 4,
        online_ft: bool = True,
        task: str = "cpc",
        num_workers: int = 4,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        pretrained: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            encoder_name: A string for any of the resnets in torchvision, or the original CPC encoder,
                or a custon nn.Module encoder
            patch_size: How big to make the image patches
            patch_overlap: How much overlap each patch should have
            online_ft: If True, enables a 1024-unit MLP to fine-tune online
            task: Which self-supervised task to use ('cpc', 'amdim', etc...)
            num_workers: number of dataloader workers
            num_classes: number of classes
            learning_rate: learning rate
            pretrained: If true, will use the weights pretrained (using CPC) on Imagenet
        """

        super().__init__()
        self.save_hyperparameters()

        self.online_evaluator = online_ft

        if pretrained:
            self.hparams.dataset = pretrained
            self.online_evaluator = True

        self.encoder = self.init_encoder()

        # info nce loss
        c, h = self.__compute_final_nb_c(patch_size)
        self.contrastive_task = CPCTask(num_input_channels=c, target_dim=64, embed_scale=0.1)

        self.z_dim = c * h * h
        self.num_classes = num_classes

        if pretrained:
            self.load_pretrained(encoder_name)

    def load_pretrained(self, encoder_name):
        available_weights = {"resnet18"}

        if encoder_name in available_weights:
            load_pretrained(self, f"CPC_v2-{encoder_name}")
        elif encoder_name not in available_weights:
            rank_zero_warn(f"{encoder_name} not yet available")

    def init_encoder(self):
        dummy_batch = torch.zeros((2, 3, self.hparams.patch_size, self.hparams.patch_size))

        encoder_name = self.hparams.encoder_name
        if encoder_name == "cpc_encoder":
            return cpc_resnet101(dummy_batch)
        return torchvision_ssl_encoder(encoder_name, return_all_feature_maps=self.hparams.task == "amdim")

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2 * 49, 3, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)

        # other encoders return a list
        if self.hparams.encoder_name != "cpc_encoder":
            dummy_batch = dummy_batch[0]

        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_patches = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_patches, nb_patches)

        return Z

    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, _, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)

        # non cpc resnets return a list
        if self.hparams.encoder_name != "cpc_encoder":
            Z = Z[0]

        # (?) -> (b, -1, nb_feats, nb_feats)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        # calculate loss
        nce_loss = self.shared_step(batch)

        # result
        self.log("train_nce_loss", nce_loss)
        return nce_loss

    def validation_step(self, batch, batch_nb):
        # calculate loss
        nce_loss = self.shared_step(batch)

        # result
        self.log("val_nce", nce_loss, prog_bar=True)
        return nce_loss

    def shared_step(self, batch):
        if isinstance(self.datamodule, STL10DataModule):
            # unlabeled batch
            batch = batch[0]

        img_1, y = batch

        # generate features
        # Latent features
        Z = self(img_1)

        # infoNCE loss
        nce_loss = self.contrastive_task(Z)
        return nce_loss

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7,
        )

        # if self.hparams.dataset in ['cifar10', 'stl10']:
        #     lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        # elif self.hparams.dataset == 'imagenet2012':
        #     lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return [opt]  # , [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        possible_resnets = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        ]
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--task", type=str, default="cpc")
        parser.add_argument("--encoder", default="cpc_encoder", type=str, choices=possible_resnets)
        # cifar10: 1e-5, stl10: 3e-5, imagenet: 4e-4
        parser.add_argument("--learning_rate", type=float, default=1e-5)

        return parser


@under_review()
def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.datamodules.ssl_imagenet_datamodule import SSLImagenetDataModule

    seed_everything(1234)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = CPC_v2.add_model_specific_args(parser)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--data_dir", default=".", type=str)
    parser.add_argument("--meta_dir", default=".", type=str, help="path to meta.bin for imagenet")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    datamodule = None
    if args.dataset == "cifar10":
        datamodule = CIFAR10DataModule.from_argparse_args(args)
        datamodule.train_transforms = CPCTrainTransformsCIFAR10()
        datamodule.val_transforms = CPCEvalTransformsCIFAR10()
        args.patch_size = 8

    elif args.dataset == "stl10":
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed
        datamodule.train_transforms = CPCTrainTransformsSTL10()
        datamodule.val_transforms = CPCEvalTransformsSTL10()
        args.patch_size = 16

    elif args.dataset == "imagenet2012":
        datamodule = SSLImagenetDataModule.from_argparse_args(args)
        datamodule.train_transforms = CPCTrainTransformsImageNet128()
        datamodule.val_transforms = CPCEvalTransformsImageNet128()
        args.patch_size = 32

    online_evaluator = SSLOnlineEvaluator(
        drop_p=0.0,
        hidden_dim=None,
        z_dim=args.hidden_mlp,
        num_classes=datamodule.num_classes,
        dataset=args.dataset,
    )
    if args.dataset == "stl10":
        # 16 GB RAM - 64
        # 32 GB RAM - 144
        args.batch_size = 144

        def to_device(batch, device):
            (_, _), (x2, y2) = batch
            x2 = x2.to(device)
            y2 = y2.to(device)
            return x2, y2

        online_evaluator.to_device = to_device

    model = CPC_v2(**vars(args))
    trainer = Trainer.from_argparse_args(args, callbacks=[online_evaluator])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
