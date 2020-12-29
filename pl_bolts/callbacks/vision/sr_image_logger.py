import torch
import torch.nn.functional as F
from pytorch_lightning import Callback

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.utils import make_grid
else:
    warn_missing_pkg("torchvision")  # pragma: no-cover


class SRImageLoggerCallback(Callback):
    def __init__(self, log_interval: int = 1000, num_samples: int = 5) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        global_step = trainer.global_step
        if global_step % self.log_interval == 0:
            hr_image, lr_image = batch
            hr_image, lr_image = hr_image.to(pl_module.device), lr_image.to(pl_module.device)
            hr_fake = pl_module(lr_image)
            lr_image = F.interpolate(lr_image, scale_factor=4)

            lr_image_grid = make_grid(lr_image[: self.num_samples], nrow=1, normalize=True)
            hr_fake_grid = make_grid(hr_fake[: self.num_samples], nrow=1, normalize=True)
            hr_image_grid = make_grid(hr_image[: self.num_samples], nrow=1, normalize=True)

            grid = torch.cat((lr_image_grid, hr_fake_grid, hr_image_grid), -1)
            title = "sr_images"
            trainer.logger.experiment.add_image(title, grid, global_step=global_step)
