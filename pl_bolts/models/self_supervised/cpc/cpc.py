import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pl_bolts.datamodules import CIFAR10DataLoaders, STL10DataLoaders
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from pl_bolts.models.self_supervised.cpc.cpc_networks import CPCResNet101
from pl_bolts.models.self_supervised.cpc import cpc_transforms
from pl_bolts.models.self_supervised.amdim.ssl_datasets import UnlabeledImagenet
from argparse import ArgumentParser
from pl_bolts import metrics
from pl_bolts.models.vision import PixelCNN
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

import math

__all__ = [
    'InfoNCE',
    'CPCV2'
]


class InfoNCE(pl.LightningModule):

    def __init__(self, num_input_channels, target_dim=64, embed_scale=0.1):
        super().__init__()
        self.target_dim = target_dim
        self.embed_scale = embed_scale

        self.target_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.pred_cnn = torch.nn.Conv2d(num_input_channels, self.target_dim, kernel_size=1)
        self.context_cnn = PixelCNN(num_input_channels)

    def compute_loss_h(self, targets, preds, i):
        b, c, h, w = targets.shape

        # (b, c, h, w) -> (num_vectors, emb_dim)
        # every vector (c-dim) is a target
        targets = targets.permute(0, 2, 3, 1).contiguous().reshape([-1, c])

        # select the future (south) targets to predict
        # selects all of the ones south of the current source
        preds_i = preds[:, :, :-(i+1), :] * self.embed_scale

        # (b, c, h, w) -> (b*w*h, c) (all features)
        # this ordering matches the targets
        preds_i = preds_i.permute(0, 2, 3, 1).contiguous().reshape([-1, self.target_dim])

        # calculate the strength scores
        logits = torch.matmul(preds_i, targets.transpose(-1, -2))

        # generate the labels
        n = b * (h - i - 1) * w
        b1 = torch.arange(n) // ((h - i - 1) * w)
        c1 = torch.arange(n) % ((h - i - 1) * w)
        labels = b1 * h * w + (i + 1) * w + c1
        labels = labels.type_as(logits).long()

        loss = nn.functional.cross_entropy(logits, labels)
        return loss

    def forward(self, Z):
        losses = []

        context = self.context_cnn(Z)
        targets = self.target_cnn(Z)

        _, _, h, w = Z.shape

        # future prediction
        preds = self.pred_cnn(context)
        for steps_to_ignore in range(h-1):
            for i in range(steps_to_ignore + 1, h):
                loss = self.compute_loss_h(targets, preds, i)
                if not torch.isnan(loss):
                    losses.append(loss)

        loss = torch.stack(losses).sum()
        return loss


class CPCV2(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.online_evaluator = self.hparams.online_ft
        self.dataset = self.get_dataset(hparams.dataset)

        # encoder network (Z vectors)
        dummy_batch = torch.zeros((2, 3, hparams.patch_size, hparams.patch_size))
        self.encoder = CPCResNet101(dummy_batch)

        # info nce loss
        c, h = self.__compute_final_nb_c(hparams.patch_size)
        self.info_nce = InfoNCE(num_input_channels=c, target_dim=64, embed_scale=0.1)

        if self.online_evaluator:
            z_dim = c * h* h
            num_classes = self.dataset.num_classes
            self.non_linear_evaluator = SSLEvaluator(
                n_input=z_dim,
                n_classes=num_classes,
                p=0.2,
                n_hidden=1024
            )

    def get_dataset(self, name):
        if name == 'cifar10':
            return CIFAR10DataLoaders(self.hparams.data_dir)
        elif name == 'stl10':
            return STL10DataLoaders(self.hparams.data_dir)

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

    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)

        # (?) -> (b, -1, nb_feats, nb_feats)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        img_1, y = batch

        # Latent features
        Z = self(img_1)

        # infoNCE loss
        nce_loss = self.info_nce(Z)
        loss = nce_loss
        log = {'train_nce_loss': nce_loss}

        # don't use the training signal, just finetune the MLP to see how we're doing downstream
        if self.online_evaluator:
            z_in = Z.detach()
            z_in = z_in.reshape(Z.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            loss = nce_loss + mlp_loss
            log['train_mlp_loss'] = mlp_loss

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
        nce_loss = self.info_nce(Z)
        result = {'val_nce': nce_loss}

        if self.online_evaluator:
            z_in = Z.reshape(Z.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, labels)
            acc = metrics.accuracy(mlp_preds, labels)
            result['mlp_acc'] = acc
            result['mlp_loss'] = mlp_loss

        return result

    def validation_epoch_end(self, outputs):
        val_nce = metrics.mean(outputs, 'val_nce')

        log = {'val_nce_loss': val_nce}
        if self.online_evaluator:
            mlp_acc = metrics.mean(outputs, 'mlp_acc')
            mlp_loss = metrics.mean(outputs, 'mlp_loss')
            log['val_mlp_acc'] = mlp_acc
            log['val_mlp_loss'] = mlp_loss

        return {'val_loss': val_nce, 'log': log}

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7
        )

        if self.hparams.dataset in ['cifar10', 'stl10']:
            lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        elif self.hparams.dataset == 'imagenet128':
            lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return [opt] # , [lr_scheduler]

    def prepare_data(self):
        self.dataset.prepare_data()

    def train_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            train_transform = cpc_transforms.CPCTransformsCIFAR10().train_transform

        elif self.hparams.dataset == 'stl10':
            stl10_transform = cpc_transforms.CPCTransformsSTL10Patches(
                patch_size=self.hparams.patch_size,
                overlap=self.hparams.patch_overlap
            )
            train_transform = stl10_transform.train_transform

        elif self.hparams.dataset == 'imagenet128':
            train_transform = cpc_transforms.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.data_dir,
                                        nb_classes=self.hparams.nb_classes,
                                        split='train',
                                        transform=train_transform)

        loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            test_transform = cpc_transforms.CPCTransformsCIFAR10().test_transform

        if self.hparams.dataset == 'stl10':
            stl10_transform = cpc_transforms.CPCTransformsSTL10Patches(
                patch_size=self.hparams.patch_size,
                overlap=self.hparams.patch_overlap
            )
            test_transform = stl10_transform.test_transform

        if self.hparams.dataset == 'imagenet128':
            train_transform = cpc_transforms.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.data_dir,
                                        nb_classes=self.hparams.nb_classes,
                                        split='val',
                                        transform=train_transform)

        loader = self.dataset.val_dataloader(self.hparams.batch_size, transforms=test_transform)

        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        from test_tube import HyperOptArgumentParser
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true')
        parser.add_argument('--dataset', type=str, default='cifar10')

        (args, _) = parser.parse_known_args()

        cifar_10 = {
            'dataset': 'cifar10',
            'depth': 10,
            'patch_size': 8,
            'batch_size': 44,
            'nb_classes': 10,
            'patch_overlap': 8 // 2,
            'lr_options': [
                # 1e-4,
                # 5e-5,
                # 1e-5,
                2e-5
                # 1e-6,
            ]
        }

        stl10 = {
            'dataset': 'stl10',
            'depth': 8,
            'patch_size': 16,
            'batch_size': 200,
            'nb_classes': 10,
            'patch_overlap': 16 // 2,
            'lr_options': [
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
            ]
        }

        imagenet128 = {
            'dataset': 'imagenet128',
            'depth': 10,
            'patch_size': 32,
            'batch_size': 60,
            'nb_classes': 1000,
            'patch_overlap': 32 // 2,
            'lr_options': [
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
            ]
        }

        DATASETS = {
            'cifar10': cifar_10,
            'stl10': stl10,
            'imagenet128': imagenet128
        }

        dataset = DATASETS[args.dataset]

        # dataset options
        parser.add_argument('--nb_classes', default=dataset['nb_classes'], type=int)
        parser.add_argument('--patch_size', default=dataset['patch_size'], type=int)
        parser.add_argument('--patch_overlap', default=dataset['patch_overlap'], type=int)

        # training params
        parser.add_argument('--batch_size', type=int, default=dataset['batch_size'])
        parser.opt_list('--learning_rate', type=float, default=0.0001, options=dataset['lr_options'], tunable=True)

        # data
        parser.add_argument('--data_dir', default=f'/home/waf251/media/falcon_kcgscratch1/datasets', type=str)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CPCV2.add_model_specific_args(parser)

    args = parser.parse_args()

    model = CPCV2(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
