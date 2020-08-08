import torch
from pytorch_lightning import Callback


class TensorboardGenerativeModelImageSampler(Callback):
    def __init__(self):
        """
        Generates images and logs to tensorboard.
        Your model must implement the forward function for generation

        Requirements::

            z = torch.rand(batch_size, latent_dim)
            img_samples = your_model(z)

        Example::

            from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

            trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
        """
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        import torchvision

        num_samples = 3
        z = torch.randn(num_samples, pl_module.hparams.latent_dim, device=pl_module.device)

        # generate images
        images = pl_module(z)

        grid = torchvision.utils.make_grid(images)
        trainer.logger.experiment.add_image('gan_images', grid, global_step=trainer.global_step)
