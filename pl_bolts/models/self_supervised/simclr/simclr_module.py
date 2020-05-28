import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.models import densenet

from pl_bolts.datamodules import CIFAR10DataLoaders, STL10DataLoaders
from pl_bolts.datamodules.ssl_imagenet_dataloaders import SSLImagenetDataLoaders
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLRDataTransform
from pl_bolts.metrics import mean
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
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    def __init__(self, dataset, data_dir, lr, wd, input_height, batch_size,
                 num_workers=0, optimizer='adam', step=30, gamma=0.5, temperature=0.5, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.input_height = input_height
        self.gamma = gamma
        self.step = step
        self.optimizer = optimizer
        self.wd = wd
        self.lr = lr
        self.temp = temperature
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.dataset = self.get_dataset(dataset)
        self.loss_func = self.init_loss()
        self.encoder = self.init_encoder()
        self.projection = self.init_projection()

    def init_loss(self):
        return nt_xent_loss

    def init_encoder(self):
        return EncoderModel()

    def init_projection(self):
        return Projection()

    def get_dataset(self, name):
        if name == 'cifar10':
            return CIFAR10DataLoaders(self.data_dir, num_workers=self.num_workers)
        elif name == 'stl10':
            return STL10DataLoaders(self.data_dir, num_workers=self.num_workers)
        elif name == 'imagenet128':
            return SSLImagenetDataLoaders(self.data_dir, num_workers=self.num_workers)
        else:
            raise FileNotFoundError(f'the {name} dataset is not supported. Subclass \'get_dataset to provide'
                                    f'your own \'')

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

    def training_step(self, batch, batch_idx):
        (img_1, img_2), y = batch
        h1, z1 = self.forward(img_1)
        h2, z2 = self.forward(img_2)

        # return h1, z1, h2, z2
        loss = self.loss_func(z1, z2, self.temp)
        logs = {'loss': loss.item()}
        return dict(loss=loss, log=logs)

    # def training_step_end(self, output_parts):
    #     h1s, z1s, h2s, z2s = output_parts
    #     rank = torch.distributed.get_rank()
    #     print(f'Rank = {rank}', [h1.shape for h1 in h1s])
    #     print(f'Rank = {rank}', [h2.shape for h2 in h2s])
    #     print(f'Rank = {rank}', [z1.shape for z1 in z1s])
    #     print(f'Rank = {rank}', [z2.shape for z2 in z2s])

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), y = batch
        h1, z1 = self.forward(img_1)
        h2, z2 = self.forward(img_2)
        loss = self.loss_func(z1, z2, self.temp)
        logs = {'val_loss': loss.item()}
        return dict(val_loss=loss, log=logs)

    def validation_epoch_end(self, outputs: list):
        val_loss = mean(outputs, 'val_loss')
        logs = dict(
            val_loss=val_loss,
        )
        return dict(val_loss=val_loss, log=logs)

    def prepare_data(self):
        self.dataset.prepare_data()

    def train_dataloader(self):
        train_transform = SimCLRDataTransform(input_height=self.input_height)
        loader = self.dataset.train_dataloader(self.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        test_transform = SimCLRDataTransform(input_height=self.input_height, test=True)
        loader = self.dataset.val_dataloader(self.batch_size, transforms=test_transform)
        return loader

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), self.lr, weight_decay=self.wd)
        elif self.optimizer == 'lars':
            optimizer = LARS(
                self.parameters(), lr=self.lr, momentum=self.mom,
                weight_decay=self.wd, eta=self.eta)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        scheduler = StepLR(
            optimizer, step_size=self.step, gamma=self.gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true')
        parser.add_argument('--dataset', type=str, default='cifar10')

        (args, _) = parser.parse_known_args()
        height = {'cifar10': 32, 'stl10':96, 'imagenet128': 224}[args.dataset]
        parser.add_argument('--input_height', type=int, default=height)

        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--expdir', type=str, default='simclrlogs')
        parser.add_argument('--optim', choices=['adam', 'lars'], default='adam')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--mom', type=float, default=0.9)
        parser.add_argument('--eta', type=float, default=0.001)
        parser.add_argument('--step', type=float, default=30)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--wd', type=float, default=0.0005)
        # Model
        parser.add_argument('--temp', type=float, default=0.5)
        parser.add_argument('--trans', type=str, default='randcrop,flip')
        return parser

    # model = SimCLR(
    #     hparams=args,
    #     encoder=EncoderModel(),
    #     projection=Projection(),
    #     loss_func=nt_xent_loss,
    #     temperature=args.temp,
    #     transform_list=list(args.trans.split(','))
    # )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SimCLR(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
