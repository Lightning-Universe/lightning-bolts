from warnings import warn

import torch
from pytorch_lightning.callbacks import Callback

try:
    import torchvision
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')


class LatentDimInterpolator(Callback):

    def __init__(self, interpolate_epoch_interval=20, range_start=-5, range_end=5, num_samples=2):
        """
        Interpolates the latent space for a model by setting all dims to zero and stepping
        through the first two dims increasing one unit at a time.

        Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)

        Example::

            from pl_bolts.callbacks import LatentDimInterpolator

            Trainer(callbacks=[LatentDimInterpolator()])

        Args:
            interpolate_epoch_interval:
            range_start: default -5
            range_end: default 5
            num_samples: default 2
        """
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples

    def on_epoch_end(self, trainer, pl_module):
        import math

        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module, latent_dim=pl_module.hparams.latent_dim)
            images = torch.cat(images, dim=0)

            num_images = (self.range_end - self.range_start) ** 2
            num_rows = int(math.sqrt(num_images))
            grid = torchvision.utils.make_grid(images, nrow=num_rows)
            str_title = f'{pl_module.__class__.__name__}_latent_space'
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module, latent_dim):
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in range(self.range_start, self.range_end, 1):
                for z2 in range(self.range_start, self.range_end, 1):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    # generate images
                    img = pl_module(z)

                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images
