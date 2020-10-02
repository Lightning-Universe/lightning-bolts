from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_examples.models.unet import UNet


class SemSegment(pl.LightningModule):

    def __init__(self,
                 data_dir: str,
                 lr: float = 0.01,
                 num_classes: int = 19,
                 num_layers: int = 5,
                 features_start: int = 64,
                 bilinear: bool = False,
                 **kwargs
    ):
        """
        Basic Semantic Segmentation Module using a UNet architecture (which can easily be substituted).

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself.
        You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        Args:
            data_dir: path to load data from
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning rate for the Adam optimizer with a CosineAnnealing scheduler
        """
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(num_classes=num_classes,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=2, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser


def cli_main():
    from pl_bolts.datamodules import KittiDataModule

    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SemSegment.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    loaders = KittiDataModule(args.data_dir).from_argparse_args(args)

    # model
    model = SemSegment(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())


if __name__ == '__main__':
    cli_main()
