"""
Adapted from: https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from pl_bolts.datamodules import CIFAR10DataLoaders, STL10DataLoaders
from pl_bolts.datamodules.ssl_imagenet_dataloaders import SSLImagenetDataLoaders
from pl_bolts.metrics import precision_at_k, mean
from pl_bolts.models.self_supervised.moco.transforms import \
    Moco2Imagenet128Transforms, Moco2CIFAR10Transforms, Moco2STL10Transforms


class MocoV2(pl.LightningModule):

    def __init__(self,
                 base_encoder='resnet50',
                 emb_dim=128,
                 num_negatives=65536,
                 encoder_momentum=0.999,
                 softmax_temperature=0.07,
                 lr=0.03,
                 momentum=0.9,
                 weight_decay=1e-4,
                 dataset='cifar10',
                 data_dir='./',
                 batch_size=256,
                 use_mlp=False,
                 *args, **kwargs):
        super().__init__()
        """
        emb_dim: feature dimension (default: 128)
        num_negatives: queue size; number of negative keys (default: 65536)
        encoder_momentum: moco momentum of updating key encoder (default: 0.999)
        softmax_temperature: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.hparams = Namespace(**{
            'emb_dim': emb_dim,
            'num_negatives': num_negatives,
            'encoder_momentum': encoder_momentum,
            'softmax_temperature': softmax_temperature,
            'use_mlp': use_mlp,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'dataset': dataset,
            'data_dir': data_dir,
            'batch_size': batch_size
        })

        self.K = num_negatives
        self.m = encoder_momentum
        self.T = softmax_temperature
        self.emb_dim = emb_dim
        self.dataset = self.get_dataset(dataset)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """

        template_model = getattr(torchvision.models, base_encoder)
        encoder_q = template_model(num_classes=self.emb_dim)
        encoder_k = template_model(num_classes=self.emb_dim)

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def get_dataset(self, name):
        if name == 'cifar10':
            dataloaders = CIFAR10DataLoaders(self.hparams.data_dir)
        elif name == 'stl10':
            dataloaders = STL10DataLoaders(self.hparams.data_dir)
        elif name == 'imagenet128':
            dataloaders = SSLImagenetDataLoaders(self.hparams.data_dir)
        else:
            raise FileNotFoundError(f'the {name} dataset is not supported. Subclass \'get_dataset to provide'
                                    f'your own \'')

        return dataloaders

    def prepare_data(self):
        self.dataset.prepare_data()

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {
            'train_loss': loss,
            'train_acc1': acc1,
            'train_acc5': acc5
        }
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        img_1, target = batch

        output = self.encoder_q(img_1)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_loss': loss,
            'val_acc1': acc1,
            'val_acc5': acc5
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {
            'val_loss': val_loss,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5
        }
        return {'val_loss': val_loss, 'log': log}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.lr,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        return optimizer

    def train_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            train_transform = Moco2CIFAR10Transforms()

        elif self.hparams.dataset == 'stl10':
            train_transform = Moco2STL10Transforms()

        elif self.hparams.dataset == 'imagenet128':
            train_transform = Moco2Imagenet128Transforms()

        loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            train_transform = Moco2CIFAR10Transforms().train_transform

        elif self.hparams.dataset == 'stl10':
            train_transform = Moco2STL10Transforms().train_transform

        elif self.hparams.dataset == 'imagenet128':
            train_transform = Moco2Imagenet128Transforms().train_transform

        loader = self.dataset.val_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        from test_tube import HyperOptArgumentParser
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_encoder', type=str, default='resnet50')
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--num_negatives', type=int, default=65536)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--lr', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--dataset', type=str, default='cifar10')
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--use_mlp', action='store_true')

        return parser


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = MocoV2.add_model_specific_args(parser)
    args = parser.parse_args()

    model = MocoV2(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True)
    trainer.fit(model)
