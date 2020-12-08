import math
import numpy as np

import torch
from pytorch_lightning.callbacks import Callback

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    import torchvision
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover


class LatentDimInterpolator(Callback):
    """
    Interpolates the latent space for a model by setting all dims to zero and stepping
    through the first two dims increasing one unit at a time.

    Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)

    Example::

        from pl_bolts.callbacks import LatentDimInterpolator

        Trainer(callbacks=[LatentDimInterpolator()])
    """

    def __init__(
        self,
        interpolate_epoch_interval: int = 20,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        num_samples: int = 2,
        normalize=False,
    ):
        """
        Args:
            interpolate_epoch_interval: default 20
            range_start: default -5
            range_end: default 5
            num_samples: default 2
            normalize: default False
        """
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module, latent_dim=pl_module.hparams.latent_dim)
            images = torch.cat(images, dim=0)

            num_images = (self.range_end - self.range_start) ** 2
            num_rows = int(math.sqrt(num_images))
            grid = torchvision.utils.make_grid(images, nrow=num_rows, normalize=self.normalize)
            str_title = f'{pl_module.__class__.__name__}_latent_space'
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module, latent_dim):
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
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
