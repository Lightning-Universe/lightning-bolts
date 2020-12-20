import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    import torchvision
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover


class TensorboardGenerativeModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation

    Requirements::

        # model must have img_dim arg
        model.img_dim = (1, 28, 28)

        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    Example::

        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(self, num_samples: int = 3) -> None:
        super().__init__()
        self.num_samples: int = num_samples

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dim = (self.num_samples, pl_module.hparams.latent_dim)  # type: ignore[union-attr]
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(images)
        str_title = f'{pl_module.__class__.__name__}_images'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
