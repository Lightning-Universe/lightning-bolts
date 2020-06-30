from argparse import ArgumentParser
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer, LightningModule, Callback
from torch.nn import functional as F

from pl_bolts.datamodules import MNISTDataModule, LightningDataModule
from pl_bolts.models.gans.basic.components import Generator, Discriminator


class GAN(LightningModule):

    def __init__(self,
                 datamodule: LightningDataModule = None,
                 latent_dim: int = 32,
                 batch_size: int = 32,
                 adam_b1: float = 0.5,
                 adam_b2: float = 0.999,
                 learning_rate: float = 0.0002,
                 data_dir: str = '',
                 num_workers: int = 8,
                 **kwargs):
        """
        Vanilla GAN implementation.

        Args:

            datamodule: the datamodule (train, val, test splits)
            latent_dim: emb dim for encoder
            batch_size: the batch size
            adam_b1: optimizer param
            adam_b2: adam params
            learning_rate: the learning rate
            data_dir: where to store data
            num_workers: data workers

        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()

        # link default data
        if datamodule is None:
            datamodule = MNISTDataModule(data_dir=self.hparams.data_dir, num_workers=self.hparams.num_workers)
        self.datamodule = datamodule
        self.img_dim = self.datamodule.size()

        # networks
        self.generator = self.init_generator(self.img_dim)
        self.discriminator = self.init_discriminator(self.img_dim)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def init_generator(self, img_dim):
        generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_dim)
        return generator

    def init_discriminator(self, img_dim):
        discriminator = Discriminator(img_shape=img_dim)
        return discriminator

    def forward(self, z):
        """
        Generates an image given input noise z

        Example::

            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)
        """
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def generator_step(self, x):
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim)
        z = z.type_as(x)

        # generate images
        self.generated_imgs = self(z)

        # ground truth result (ie: all real)
        real = torch.ones(x.size(0), 1)
        real = real.type_as(x)
        g_loss = self.generator_loss(real)

        tqdm_dict = {'g_loss': g_loss}
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def generator_loss(self, real):
        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), real)
        return g_loss

    def discriminator_loss(self, x):
        # how well can it label as real?
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)

        real_loss = self.adversarial_loss(self.discriminator(x), valid)

        # how well can it label as fake?
        fake = torch.zeros(x.size(0), 1)
        fake = fake.type_as(x)

        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({
            'loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        self.last_imgs = x

        # train generator
        if optimizer_idx == 0:
            return self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            return self.discriminator_step(x)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        adam_b1 = self.hparams.adam_b1
        adam_b2 = self.hparams.adam_b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(adam_b1, adam_b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(adam_b1, adam_b2))
        return [opt_g, opt_d], []

    def prepare_data(self):
        self.datamodule.prepare_data()

    def train_dataloader(self):
        return self.datamodule.train_dataloader(self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument('--adam_b1', type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--adam_b2', type=float, default=0.999,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--latent_dim', type=int, default=100,
                            help="generator embedding dim")
        parser.add_argument('--batch_size', type=int, default=64, help="size of the batches")
        parser.add_argument('--num_workers', type=int, default=8, help="num dataloader workers")
        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--dataset', type=str, default='mnist')

        return parser


class ImageGenerator(Callback):

    def on_epoch_end(self, trainer, pl_module):
        import torchvision

        num_samples = 3
        z = torch.randn(num_samples, pl_module.hparams.latent_dim)

        # generate images
        images = pl_module(z)

        grid = torchvision.utils.make_grid(images)
        trainer.logger.experiment.add_image('gan_images', grid, 0)


if __name__ == '__main__':
    from pl_bolts.datamodules import ImagenetDataModule

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = GAN.add_model_specific_args(parser)
    parser = ImagenetDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    datamodule = None
    if args.dataset == 'imagenet2012' or args.pretrained:
        datamodule = ImagenetDataModule.from_argparse_args(args)

    gan = GAN(**vars(args), datamodule=datamodule)
    trainer = Trainer.from_argparse_args(args, callbacks=[ImageGenerator()])
    trainer.fit(gan)
