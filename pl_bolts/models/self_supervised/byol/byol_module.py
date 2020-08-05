from copy import deepcopy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.optimizers.layer_adaptive_scaling import LARS
from pl_bolts.models.self_supervised.byol.models import SiameseArm
from pl_bolts.callbacks.self_supervised import BYOLMAWeightUpdate


class BYOL(pl.LightningModule):
    def __init__(self,
                 datamodule: pl.LightningDataModule = None,
                 data_dir: str = './',
                 learning_rate: float = 0.00006,
                 weight_decay: float = 0.0005,
                 input_height: int = 32,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 optimizer: str = 'lars',
                 lr_sched_step: float = 30.0,
                 lr_sched_gamma: float = 0.5,
                 lars_momentum: float = 0.9,
                 lars_eta: float = 0.001,
                 loss_temperature: float = 0.5,
                 **kwargs):
        """
        PyTorch Lightning implementation of `Bring Your Own Latent Space (BYOL)
        <https://arxiv.org/pdf/2006.07733.pdf.>`_

        Paper authors: Jean-Bastien Grill ,Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
        Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
        Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

        Model implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        .. warning:: Work in progress. This implementation is still being verified.

        TODOs:
            - add cosine scheduler
            - verify on CIFAR-10
            - verify on STL-10
            - pre-train on imagenet

        Example:

            >>> from pl_bolts.models.self_supervised import BYOL
            ...
            >>> model = BYOL()

        Train::

            trainer = Trainer()
            trainer.fit(model)

        CLI command::

            # cifar10
            python byol_module.py --gpus 1

            # imagenet
            python byol_module.py
                --gpus 8
                --dataset imagenet2012
                --data_dir /path/to/imagenet/
                --meta_dir /path/to/folder/with/meta.bin/
                --batch_size 32

        Args:
            datamodule: The datamodule
            data_dir: directory to store data
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            optimizer: optimizer name
            lr_sched_step: step for learning rate scheduler
            lr_sched_gamma: gamma for learning rate scheduler
            lars_momentum: the mom param for lars optimizer
            lars_eta: for lars optimizer
            loss_temperature: float = 0.
        """
        super().__init__()
        self.save_hyperparameters()

        # init default datamodule
        if datamodule is None:
            datamodule = CIFAR10DataModule(data_dir, num_workers=num_workers, batch_size=batch_size)
            datamodule.train_transforms = SimCLRTrainDataTransform(input_height)
            datamodule.val_transforms = SimCLREvalDataTransform(input_height)

        self.datamodule = datamodule

        self.online_network = SiameseArm()
        self.target_network = deepcopy(self.online_network)

        self.weight_callback = BYOLMAWeightUpdate()

    def on_batch_end(self):
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_batch_end(self.trainer, self)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        (img_1, img_2), y = batch

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = F.mse_loss(h1, z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        loss_b = F.mse_loss(h1, z2)

        # final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.TrainResult(minimize=total_loss)
        result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return result

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.EvalResult(early_stop_on=total_loss, checkpoint_on=total_loss)
        result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return result

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.hparams.learning_rate)
        # TODO: add scheduler - cosine decay
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, imagenet2012, stl10')

        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--optimizer', choices=['adam', 'lars'], default='lars')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--lars_momentum', type=float, default=0.9)
        parser.add_argument('--lars_eta', type=float, default=0.001)
        parser.add_argument('--lr_sched_step', type=float, default=30, help='lr scheduler step')
        parser.add_argument('--lr_sched_gamma', type=float, default=0.5, help='lr scheduler step')
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        # Model
        parser.add_argument('--loss_temperature', type=float, default=0.5)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    datamodule = None
    if args.dataset == 'stl10':
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed

        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    elif args.dataset == 'imagenet2012':
        datamodule = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    model = BYOL(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(args, max_steps=10000)
    trainer.fit(model)
