from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.models import densenet
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.optimizers import LARS
from pl_bolts.datamodules import CIFAR10DataLoaders, STL10DataLoaders
from pl_bolts.datamodules.ssl_imagenet_dataloaders import SSLImagenetDataLoaders
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLRDataTransform


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet.densenet121(densenet.ModelCfg(), num_classes=1)
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
    def __init__(self, hparams, encoder, projection, loss_func, temperature, transform_list):
        super().__init__()

        self.dataset = self.get_dataset(hparams.dataset)
        self.hparams = hparams
        self.encoder = encoder
        self.projection = projection
        self.loss_func = loss_func
        self.temp = temperature
        self.transform_list = transform_list

    def get_dataset(self, name):
        if name == 'cifar10':
            return CIFAR10DataLoaders(self.hparams.data_dir, num_workers=self.hparams.num_workers)
        elif name == 'stl10':
            return STL10DataLoaders(self.hparams.data_dir, num_workers=self.hparams.num_workers)
        elif name == 'imagenet128':
            return SSLImagenetDataLoaders(self.hparams.data_dir, num_workers=self.hparams.num_workers)
        else:
            raise FileNotFoundError(f'the {name} dataset is not supported. Subclass \'get_dataset to provide'
                                    f'your own \'')

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

    def training_step(self, batch, batch_idx):
        h1, z1 = self.forward(batch['PA'])
        h2, z2 = self.forward(batch['PA2'])
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
        h1, z1 = self.forward(batch['PA'])
        h2, z2 = self.forward(batch['PA2'])
        loss = self.loss_func(z1, z2, self.temp)
        logs = {'val_loss': loss.item()}
        self.lossmeter.add(loss.item(), h1.shape[0])
        return dict(loss=loss, log=logs)

    def validation_epoch_end(self, outputs: list):
        logs = dict(
            val_loss=self.lossmeter.mean,
        )
        print('\nLogs: ', logs)
        return dict(val_loss=self.lossmeter.mean, log=logs)

    def prepare_data(self):
        self.dataset.prepare_data()

    def train_dataloader(self):
        train_transform = SimCLRDataTransform(input_height=self.input_height)
        loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        test_transform = SimCLRDataTransform(input_height=self.input_height).test_transform
        loader = self.dataset.val_dataloader(self.hparams.batch_size, transforms=test_transform)
        return loader

    def configure_optimizers(self):
        if self.hparams.optim == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optim == 'lars':
            optimizer = LARS(
                self.parameters(), lr=self.hparams.lr, momentum=self.hparams.mom,
                weight_decay=self.hparams.wd, eta=self.hparams.eta)
        else:
            raise ValueError(f'Invalid optimizer: {self.hparams.optim}')
        scheduler = StepLR(
            optimizer, step_size=self.hparams.step, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true')
        parser.add_argument('--dataset', type=str, default='cifar10')

        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--expdir', type=str, default='simclrlogs')
        parser.add_argument('--dataset', type=str, default='CheXpert-v1.0')
        parser.add_argument('--optim', choices=['adam', 'lars'], default='adam')
        parser.add_argument('--bsz', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--mom', type=float, default=0.9)
        parser.add_argument('--eta', type=float, default=0.001)
        parser.add_argument('--step', type=float, default=30)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--wd', type=float, default=0.0005)
        parser.add_argument('--gpus', type=int, default=8)
        # Model
        parser.add_argument('--temp', type=float, default=0.5)
        parser.add_argument('--trans', type=str, default='randcrop,flip')
        return parser


def main(args):
    print('Args:', args)
    model = SimCLR(
        hparams=args,
        encoder=EncoderModel(),
        projection=Projection(),
        loss_func=nt_xent_loss,
        temperature=args.temp,
        transform_list=list(args.trans.split(','))
    )
    checkpoint = ModelCheckpoint(
        filepath=Path(args.expdir) / 'checkpoints', save_top_k=100)
    trainer = pl.Trainer(
        distributed_backend='ddp',
        gpus=args.gpus,
        default_save_path=args.expdir,
        checkpoint_callback=checkpoint
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    main(args)