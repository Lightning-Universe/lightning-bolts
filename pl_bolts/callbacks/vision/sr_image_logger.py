from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.utils import make_grid
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class SRImageLoggerCallback(Callback):
    """Logs low-res, generated high-res, and ground truth high-res images to TensorBoard Your model must implement
    the ``forward`` function for generation.

    Requirements::

        # model forward must work generating high-res from low-res image
        hr_fake = pl_module(lr_image)

    Example::

        from pl_bolts.callbacks import SRImageLoggerCallback

        trainer = Trainer(callbacks=[SRImageLoggerCallback()])
    """

    def __init__(self, log_interval: int = 1000, scale_factor: int = 4, num_samples: int = 5) -> None:
        """
        Args:
            log_interval: Number of steps between logging. Default: ``1000``.
            scale_factor: Scale factor used for downsampling the high-res images. Default: ``4``.
            num_samples: Number of images of displayed in the grid. Default: ``5``.
        """
        super().__init__()
        self.log_interval = log_interval
        self.scale_factor = scale_factor
        self.num_samples = num_samples

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        global_step = trainer.global_step
        if global_step % self.log_interval == 0:
            hr_image, lr_image = batch
            hr_image, lr_image = hr_image.to(pl_module.device), lr_image.to(pl_module.device)
            hr_fake = pl_module(lr_image)
            lr_image = F.interpolate(lr_image, scale_factor=self.scale_factor)

            lr_image_grid = make_grid(lr_image[: self.num_samples], nrow=1, normalize=True)
            hr_fake_grid = make_grid(hr_fake[: self.num_samples], nrow=1, normalize=True)
            hr_image_grid = make_grid(hr_image[: self.num_samples], nrow=1, normalize=True)

            grid = torch.cat((lr_image_grid, hr_fake_grid, hr_image_grid), -1)
            title = "sr_images"
            trainer.logger.experiment.add_image(title, grid, global_step=global_step)
