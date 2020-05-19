import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from pl_bolts.metrics.self_supervised.losses import AMDIMLossNCE
from pl_bolts.models.self_supervised.amdim.networks import AMDIMEncoder
from pl_bolts.models.self_supervised.amdim.datasets import AMDIMPretraining


class AMDIM(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        dummy_batch = torch.zeros((2, 3, hparams.image_height, hparams.image_height))

        self.encoder = AMDIMEncoder(
            dummy_batch,
            num_channels=3,
            ndf=hparams.ndf,
            n_rkhs=hparams.n_rkhs,
            n_depth=hparams.n_depth,
            encoder_size=hparams.image_height,
            use_bn=hparams.use_bn
        )
        self.encoder.init_weights()

        # the loss has learnable parameters
        self.nce_loss = AMDIMLossNCE(tclip=self.hparams.tclip)

        self.tng_split = None
        self.val_split = None

    def forward(self, img_1, img_2):
        # feats for img 1
        # r1 = last layer out
        # r5 = last layer with (b, c, 5, 5) size
        # r7 = last layer with (b, c, 7, 7) size
        r1_x1, r5_x1, r7_x1 = self.encoder(img_1)

        # feats for img 2
        r1_x2, r5_x2, r7_x2 = self.encoder(img_2)

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
            'r1_x1': r1_x1,
            'r5_x1': r5_x1,
            'r7_x1': r7_x1,
            'r1_x2': r1_x2,
            'r5_x2': r5_x2,
            'r7_x2': r7_x2,
        }

        return result

    def training_end(self, outputs):
        r1_x1 = outputs['r1_x1']
        r5_x1 = outputs['r5_x1']
        r7_x1 = outputs['r7_x1']
        r1_x2 = outputs['r1_x2']
        r5_x2 = outputs['r5_x2']
        r7_x2 = outputs['r7_x2']

        # ------------------
        # NCE LOSS
        loss_1t5, loss_1t7, loss_5t5, lgt_reg = self.nce_loss(r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2)
        unsupervised_loss = loss_1t5 + loss_1t7 + loss_5t5 + lgt_reg

        # ------------------
        # FULL LOSS
        total_loss = unsupervised_loss

        tensorboard_logs = {'train_nce_loss': total_loss}
        result = {
            'loss': total_loss,
            'log': tensorboard_logs
        }

        return result

    def validation_step(self, batch, batch_nb):
        [img_1, img_2], labels = batch

        # generate features
        r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2 = self.forward(img_1, img_2)

        # NCE LOSS
        loss_1t5, loss_1t7, loss_5t5, lgt_reg = self.nce_loss(r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2)
        unsupervised_loss = loss_1t5 + loss_1t7 + loss_5t5 + lgt_reg

        result = {
            'val_nce': unsupervised_loss
        }
        return result

    def validation_epoch_end(self, outputs):
        val_nce = 0
        for output in outputs:
            val_nce += output['val_nce']

        val_nce = val_nce / len(outputs)
        tensorboard_logs = {'val_nce': val_nce}
        return {'val_loss': val_nce, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7
        )

        if self.hparams.dataset_name in ['CIFAR10', 'stl_10', 'CIFAR100']:
            lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        else:
            lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return opt  # [opt], [lr_scheduler]

    def train_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            dataset = AMDIMPretraining.cifar10_train(self.hparams.data_dir)

        if self.hparams.dataset_name == 'stl_10':
            self.tng_split, self.val_split = AMDIMPretraining.stl_train(self.hparams.data_dir)
            dataset = self.tng_split

        if self.hparams.dataset_name == 'imagenet_128':
            dataset = AMDIMPretraining.imagenet_train(self.hparams.data_dir, self.hparams.nb_classes)

        # LOADER
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
            dataset = AMDIMPretraining.cifar10_val(self.hparams.data_dir)

        if self.hparams.dataset_name == 'stl_10':
            dataset = self.val_split

        if self.hparams.dataset_name == 'imagenet_128':
            dataset = AMDIMPretraining.imagenet_val(self.hparams.data_dir, self.hparams.nb_classes)

        # LOADER
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
        cf_root_lr = 2e-4
        cifar_10 = {
            'dataset_name': 'CIFAR10',
            'ndf': 320,
            'n_rkhs': 1280,
            'depth': 10,
            'image_height': 32,
            'batch_size': 200,
            'nb_classes': 10,
            'lr_options': [
                cf_root_lr * 32,
                cf_root_lr * 16,
                cf_root_lr * 8,
                cf_root_lr * 4,
                cf_root_lr * 2,
                cf_root_lr,
                cf_root_lr * 1 / 2,
                cf_root_lr * 1 / 4,
                cf_root_lr * 1 / 8,
                cf_root_lr * 1 / 16,
                cf_root_lr * 1 / 32,
            ]
        }

        # stl-10
        stl_root_lr = 2e-4
        stl_10 = {
            'dataset_name': 'stl_10',
            'ndf': 192,
            'n_rkhs': 1536,
            'depth': 8,
            'image_height': 64,
            'batch_size': 200,
            'nb_classes': 10,
            'lr_options': [
                stl_root_lr * 32,
                stl_root_lr * 16,
                stl_root_lr * 8,
                stl_root_lr * 4,
                stl_root_lr * 2,
                stl_root_lr,
                stl_root_lr * 1 / 2,
                stl_root_lr * 1 / 4,
                stl_root_lr * 1 / 8,
                stl_root_lr * 1 / 16,
                stl_root_lr * 1 / 32,
            ]
        }

        imagenet_root_lr = 2e-4
        imagenet_128 = {
            'dataset_name': 'imagenet_128',
            'ndf': 320,
            'n_rkhs': 2560,
            'depth': 10,
            'image_height': 128,
            'batch_size': 200,
            'nb_classes': 1000,
            'lr_options': [
                imagenet_root_lr * 32,
                imagenet_root_lr * 16,
                imagenet_root_lr * 8,
                imagenet_root_lr * 4,
                imagenet_root_lr * 2,
                imagenet_root_lr,
                imagenet_root_lr * 1 / 2,
                imagenet_root_lr * 1 / 4,
                imagenet_root_lr * 1 / 8,
                imagenet_root_lr * 1 / 16,
                imagenet_root_lr * 1 / 32,
            ]
        }

        imagenet_128_large = {
            'dataset_name': 'imagenet_128',
            'ndf': 320,
            'n_rkhs': 2560,
            'depth': 10,
            'image_height': 128,
            'batch_size': 200,
            'nb_classes': 1000,
            'lr_options': [
                imagenet_root_lr * 32,
                imagenet_root_lr * 16,
                imagenet_root_lr * 8,
                imagenet_root_lr * 4,
                imagenet_root_lr * 2,
                imagenet_root_lr,
                imagenet_root_lr * 1 / 2,
                imagenet_root_lr * 1 / 4,
                imagenet_root_lr * 1 / 8,
                imagenet_root_lr * 1 / 16,
                imagenet_root_lr * 1 / 32,
            ]
        }

        # dataset = cifar_10
        # dataset = stl_10
        dataset = cifar_10

        # dataset options
        parser.add_argument('--nb_classes', default=dataset['nb_classes'], type=int)

        # network params
        parser.add_argument('--tclip', type=float, default=20.0, help='soft clipping range for NCE scores')
        parser.add_argument('--use_bn', type=int, default=0)
        parser.add_argument('--ndf', type=int, default=dataset['ndf'], help='feature width for encoder')
        parser.add_argument('--n_rkhs', type=int, default=dataset['n_rkhs'],
                            help='number of dimensions in fake RKHS embeddings')
        parser.add_argument('--n_depth', type=int, default=dataset['depth'])
        parser.add_argument('--image_height', type=int, default=dataset['image_height'])

        # trainin params
        resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                   'wide_resnet50_2', 'wide_resnet101_2']
        parser.add_argument('--dataset_name', type=str, default=dataset['dataset_name'])
        parser.add_argument('--batch_size', type=int, default=dataset['batch_size'],
                            help='input batch size (default: 200)')
        parser.add_argument('--learning_rate', type=float, default=0.0002)

        # data
        parser.add_argument('--data_dir', default=os.getcwd(), type=str)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AMDIM.add_model_specific_args(parser)

    args = parser.parse_args()

    model = AMDIM(args)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
