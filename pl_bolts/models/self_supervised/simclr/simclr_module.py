import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.models import densenet

from pl_bolts import metrics
from pl_bolts.datamodules import get_datamodule
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.metrics import mean
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLRDataTransform
from pl_bolts.optimizers.layer_adaptive_scaling import LARS


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet.densenet121(pretrained=False, num_classes=1)
        del self.model.classifier

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


class Projection(nn.Module):
    def __init__(self, input_dim=1024, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    def __init__(self,
                 dataset: str = 'cifar10',
                 data_dir: str = '',
                 learning_rate: float = 0.00006,
                 weight_decay: float = 0.0005,
                 input_height: int = 32,
                 batch_size: int = 128,
                 online_ft: bool = False,
                 num_workers: int = 4,
                 optimizer: str = 'adam',
                 step: int = 30,
                 lr_sched_gamma: float = 0.5,
                 loss_temperature: float = 0.5,
                 **kwargs):
        """
        PyTorch Lightning implementation of `SIMCLR <https://arxiv.org/abs/2002.05709.>`_

        Paper authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.

        Model implemented by:

            - `William Falcon <https://github.com/williamFalcon>`_
            - `Tullie Murrel <https://github.com/tullie>`_

        Example:

            >>> from pl_bolts.models.self_supervised import SimCLR
            ...
            >>> model = SimCLR()

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            dataset: dataset name
            data_dir: directory to store data
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            online_ft: whether to tune online or not
            num_workers: number of workers
            optimizer: optimizer name
            lr_sched_step: step for learning rate scheduler
            lr_sched_gamma: gamma for learning rate scheduler
            loss_temperature: float = 0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.online_evaluator = online_ft

        self.dataset = self.get_dataset(dataset)
        self.loss_func = self.init_loss()
        self.encoder = self.init_encoder()
        self.projection = self.init_projection()

        if self.online_evaluator:
            z_dim = self.projection.output_dim
            num_classes = self.dataset.num_classes
            self.non_linear_evaluator = SSLEvaluator(
                n_input=z_dim,
                n_classes=num_classes,
                p=0.2,
                n_hidden=1024
            )

    def init_loss(self):
        return nt_xent_loss

    def init_encoder(self):
        return EncoderModel()

    def init_projection(self):
        return Projection()

    def get_dataset(self, name):
        extra = dict(meta_root=self.hparams.meta_root) if name == 'imagenet2012' else {}
        return get_datamodule(name, data_dir=self.hparams.data_dir, num_workers=self.hparams.num_workers, **extra)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

    def training_step(self, batch, batch_idx):
        if self.hparams.dataset == 'stl10':
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        h1, z1 = self.forward(img_1)
        h2, z2 = self.forward(img_2)

        # return h1, z1, h2, z2
        loss = self.loss_func(z1, z2, self.hparams.loss_temperature)
        log = {'train_ntx_loss': loss}

        # don't use the training signal, just finetune the MLP to see how we're doing downstream
        if self.online_evaluator:
            if self.hparams.dataset == 'stl10':
                (img_1, img_2), y = labeled_batch

            with torch.no_grad():
                h1, z1 = self.forward(img_1)

            # just in case... no grads into unsupervised part!
            z_in = z1.detach()

            z_in = z_in.reshape(z_in.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            loss = loss + mlp_loss
            log['train_mlp_loss'] = mlp_loss

        result = {
            'loss': loss,
            'log': log
        }

        return result

    def validation_step(self, batch, batch_idx):
        if self.hparams.dataset == 'stl10':
            labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), y = batch
        h1, z1 = self.forward(img_1)
        h2, z2 = self.forward(img_2)
        loss = self.loss_func(z1, z2, self.hparams.loss_temperature)
        result = {'val_loss': loss}

        if self.online_evaluator:
            if self.hparams.dataset == 'stl10':
                (img_1, img_2), y = labeled_batch
                h1, z1 = self.forward(img_1)

            z_in = z1.reshape(z1.size(0), -1)
            mlp_preds = self.non_linear_evaluator(z_in)
            mlp_loss = F.cross_entropy(mlp_preds, y)
            acc = metrics.accuracy(mlp_preds, y)
            result['mlp_acc'] = acc
            result['mlp_loss'] = mlp_loss

        return result

    def validation_epoch_end(self, outputs: list):
        val_loss = mean(outputs, 'val_loss')

        log = dict(
            val_loss=val_loss,
        )

        progress_bar = {}
        if self.online_evaluator:
            mlp_acc = mean(outputs, 'mlp_acc')
            mlp_loss = mean(outputs, 'mlp_loss')
            log['val_mlp_acc'] = mlp_acc
            log['val_mlp_loss'] = mlp_loss
            progress_bar['val_acc'] = mlp_acc

        return dict(val_loss=val_loss, log=log, progress_bar=progress_bar)

    def prepare_data(self):
        self.dataset.prepare_data()

    def train_dataloader(self):
        train_transform = SimCLRDataTransform(input_height=self.hparams.input_height)

        if self.hparams.dataset == 'stl10':
            loader = self.dataset.train_dataloader_mixed(self.hparams.batch_size, transforms=train_transform)
        else:
            loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        test_transform = SimCLRDataTransform(input_height=self.hparams.input_height, test=True)

        if self.hparams.dataset == 'stl10':
            loader = self.dataset.val_dataloader_mixed(self.hparams.batch_size, transforms=test_transform)
        else:
            loader = self.dataset.val_dataloader(self.hparams.batch_size, transforms=test_transform)
        return loader

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'lars':
            optimizer = LARS(
                self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.mom,
                weight_decay=self.hparams.weight_decay, eta=self.hparams.eta)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        scheduler = StepLR(
            optimizer, step_size=self.hparams.lr_sched_step, gamma=self.hparams.lr_sched_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, imagenet, stl10')

        (args, _) = parser.parse_known_args()
        height = {'cifar10': 32, 'stl10': 96, 'imagenet2012': 224}[args.dataset]
        parser.add_argument('--input_height', type=int, default=height)

        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--optimizer', choices=['adam', 'lars'], default='lars')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.00006)
        parser.add_argument('--mom', type=float, default=0.9)
        parser.add_argument('--eta', type=float, default=0.001)
        parser.add_argument('--lr_sched_step', type=float, default=30, help='lr scheduler step')
        parser.add_argument('--lr_sched_gamma', type=float, default=0.5, help='lr scheduler step')
        parser.add_argument('--weight_decay', type=float, default=0.0005)
        # Model
        parser.add_argument('--loss_temperature', type=float, default=0.5)
        parser.add_argument('--num_workers', default=0, type=int)
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SimCLR(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, overfit_batches=1)
    trainer.fit(model)
