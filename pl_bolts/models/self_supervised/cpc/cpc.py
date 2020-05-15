import torch
import torch.optim as optim
from torch import nn
from torchvision.datasets import STL10, CIFAR10
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from pl_bolts.metrics.self_supervised.losses import CPCV2LossInfoNCE
from pl_bolts.models.self_supervised.cpc.cpc_networks import CPCResNet101, MaskedConv2d
from pl_bolts.models.self_supervised.cpc import cpc_transforms
from pl_bolts.models.self_supervised.amdim.ssl_datasets import UnlabeledImagenet
from argparse import ArgumentParser
from pl_bolts import metrics
from pl_bolts.models.vision import PixelCNN

import math
pl.seed_everything(123)


class InfoNCE(pl.LightningModule):

    def __init__(self, num_input_channels, target_dim=64, embed_scale=0.1):
        super().__init__()
        self.target_dim = target_dim
        self.embed_scale = embed_scale

        self.target_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.pred_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.context_cnn = PixelCNN(num_input_channels)

    def forward(self, Z, steps_to_ignore=2, steps_to_predict=3):
        loss = 0.0
        import pdb; pdb.set_trace()

        # generate the context vectors
        C = self.context_cnn(Z)

        # generate targets
        targets = self.target_cnn(C)
        b, c, h, w = targets.shape
        targets = targets.permute(0, 2, 3, 1).contiguous().reshape([-1, c])

        for i in range(steps_to_ignore, steps_to_predict):
            n = b * (h - i - 1) * w

            preds_i = self.pred_cnn(C)
            preds_i = preds_i[:, :, :-(i + 1), :] * self.embed_scale
            preds_i = preds_i.permute(0, 2, 3, 1).contiguous().reshape([-1, self.target_dim])

            logits = torch.mm(preds_i, targets.transpose(1, 0))

            b1 = torch.arange(n, device=self.device) // ((h - i - 1) * w)
            c1 = torch.arange(n, device=self.device) % ((h - i - 1) * w)
            labels = b1 * h * w + (i + 1) * w + c1
            import pdb;pdb.set_trace()
            loss += nn.functional.cross_entropy(logits, labels)

        return loss


class CPCV2(pl.LightningModule):

    # ------------------------------
    # INIT
    # ------------------------------
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # encoder network (Z vectors)
        dummy_batch = torch.zeros((2, 3, hparams.patch_size, hparams.patch_size))
        self.encoder = CPCResNet101(dummy_batch)

        # info nce loss
        c, h = self.__compute_final_nb_c(hparams.patch_size)
        self.info_nce = InfoNCE(num_input_channels=c, target_dim=64, embed_scale=0.1)
        self.info_nce.device = self.device

        self.tng_split = None
        self.val_split = None

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2*49, 3, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)
        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)

        return Z

    # ------------------------------
    # FWD
    # ------------------------------
    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        img_1, _ = batch

        # Latent features
        Z = self(img_1)

        # infoNCE loss
        loss = self.info_nce(Z)

        log = {'val_nce_loss': loss}
        result = {
            'loss': loss,
            'log': log
        }

        return result

    def validation_step(self, batch, batch_nb):
        img_1, labels = batch

        # generate features
        # Latent features
        Z = self(img_1)

        # infoNCE loss
        loss = self.info_nce(Z)

        result = {
            'val_nce': loss
        }
        return result

    def validation_epoch_end(self, outputs):
        val_nce = metrics.mean(outputs, 'val_nce')

        log = {'val_nce_loss': val_nce}
        return {'val_loss': val_nce, 'log': log}

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7
        )

        if self.hparams.dataset_name == 'CIFAR10': # Dataset.C100, Dataset.STL10
            lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        else:
            lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return [opt], [lr_scheduler]

    def prepare_data(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = cpc_transforms.CPCTransformsC10()
            CIFAR10(root=self.hparams.data_dir, train=True, transform=train_transform, download=True)
            CIFAR10(root=self.hparams.data_dir, train=False, transform=train_transform, download=True)

    def train_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = cpc_transforms.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=train_transform, download=False)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

        if self.hparams.dataset_name == 'stl_10':
            train_transform = cpc_transforms.CPCTransformsSTL10Patches(patch_size=self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = STL10(root=self.hparams.data_dir, split='unlabeled', transform=train_transform, download=True)

            self.tng_split, self.val_split = random_split(dataset, [95000, 5000])

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = cpc_transforms.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.data_dir,
                                        nb_classes=self.hparams.nb_classes,
                                        split='train',
                                        transform=train_transform)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

    def val_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = cpc_transforms.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=train_transform, download=False)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

        if self.hparams.dataset_name == 'stl_10':
            loader = DataLoader(
                dataset=self.val_split,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = cpc_transforms.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.data_dir,
                                        nb_classes=self.hparams.nb_classes,
                                        split='val',
                                        transform=train_transform)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
            )

            return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # CIFAR 10
        patch_size = 8
        cifar_10 = {
            'dataset_name': 'CIFAR10',
            'depth': 10,
            'patch_size': patch_size,
            'batch_size': 200,
            'nb_classes': 10,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                2e-5,
            ]
        }

        # stl-10
        patch_size = 16
        stl_10 = {
            'dataset_name': 'stl_10',
            'depth': 8,
            'patch_size': patch_size,
            'batch_size': 200,
            'nb_classes': 10,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                # 2e-7,
                # 2e-4*(1/4),
                # 2e-4*(1/2),
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
                # 2e-4*2,
                # 3e-4,
                # 2e-4*3,
                # 2e-4*4,
                # 8e-4,
                # 2e-4 * 8, 2e-4 * 16
            ]
        }

        # imagenet_128
        patch_size = 32
        imagenet_128 = {
            'dataset_name': 'imagenet_128',
            'depth': 10,
            'patch_size': patch_size,
            'batch_size': 60,
            'nb_classes': 1000,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
            ]
        }

        dataset = cifar_10
        # dataset = stl_10
        # dataset = imagenet_128

        # dataset options
        parser.add_argument('--nb_classes', default=dataset['nb_classes'], type=int)
        parser.add_argument('--patch_size', default=dataset['patch_size'], type=int)
        parser.add_argument('--patch_overlap', default=dataset['patch_overlap'], type=int)

        # trainin params
        resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
        parser.add_argument('--dataset_name', type=str, default=dataset['dataset_name'])
        parser.add_argument('--batch_size', type=int, default=dataset['batch_size'])
        parser.add_argument('--learning_rate', type=float, default=0.0001)

        # data
        parser.add_argument('--data_dir', default=f'./', type=str)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CPCV2.add_model_specific_args(parser)

    args = parser.parse_args()

    model = CPCV2(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
