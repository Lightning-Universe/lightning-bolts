import os
from argparse import ArgumentParser

from collections import OrderedDict
import torch
from pytorch_lightning import Trainer, LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from pl_bolts.models.gans.basic.components import Generator, Discriminator


class BasicGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # networks
        self.generator = self.init_generator(self.hparams)
        self.discriminator = self.init_discriminator(self.hparams)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def init_generator(self, hparams):
        image_shape = (hparams.input_channels, hparams.input_width, hparams.input_height)
        generator = Generator(latent_dim=hparams.latent_dim, img_shape=image_shape)
        return generator

    def init_discriminator(self, hparams):
        image_shape = (hparams.input_channels, hparams.input_width, hparams.input_height)
        discriminator = Discriminator(img_shape=image_shape)
        return discriminator

    def forward(self, z):
        """
        Allows infernce to be about generating images
        x = gan(z)
        :param z:
        :return:
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
        fake = fake.type_as(fake)

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
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.add_argument('--input_width', type=int, default=28,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_channels', type=int, default=1,
                            help='num channels')
        parser.add_argument('--input_height', type=int, default=28,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument('--b1', type=float, default=0.5,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--b2', type=float, default=0.999,
                            help="adam: decay of first order momentum of gradient")
        parser.add_argument('--latent_dim', type=int, default=100,
                            help="dimensionality of the latent space")
        parser.add_argument('--batch_size', type=int, default=64, help="size of the batches")

        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BasicGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    gan = BasicGAN(args)
    trainer = Trainer()
    trainer.fit(gan)
