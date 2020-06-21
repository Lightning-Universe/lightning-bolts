Introduction Guide
==================
Lightning Bolts is a collection of Models, Callbacks and other goodies implemented in PyTorch Lightning.

Bolts models are designed to bootstrap research or to be used in production. Here are ways in which bolts can be used

1. As a feature extractor for production and research systems.
2. Subclass and override to try new research ideas.
3. Fully pre-built models that can be fine-tuned on your data.
3. Fully contained algorithms that can be trained on your data.
4. Fully compatible with GPUs, TPUs, 16-bit precision, etc because of PyTorch Lightning.
5. Can be used as a stand-alone `torch.nn.Module`.

--------------------

Use as a feature extractor
--------------------------
For certain projects that require an architecture you could use this as
a module inside the larger system.

Most models have pretrained weights (usually on Imagenet).

Example::

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (pretrained on Imagenet)
    pretrained_model = VAE(pretrained=True)
    pretrained_model.freeze()

Or load your own weights after training on your own data

Example::

    from pl_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    # train
    model = VAE()
    trainer = pl.Trainer()
    trainer.fit(model)

    # feature extractor
    pretrained_model = VAE.load_from_checkpoint(PATH)
    pretrained_model.freeze()

Use for fine-tuning
-------------------
Can fine-tune on your own data. Either for stand-alone PyTorch

Example::

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (not frozen)
    pretrained_model = VAE.load_from_checkpoint(PATH)

Or in a Lightning Module

Example::

    class YourResearchModel(pl.LightningModule):
        def __init__(self):

            # pretrained VAE
            self.vae = VAE.load_from_checkpoint(PATH)
            self.vae.freeze()

            self.some_other_model = MyModel()

        def forward(self, z):
            # unfreeze at some point
            if self.current_epoch == 10:
                self.vae.unfreeze()

            # generate a sample from z ~ N(0,1)
            x = self.vae(z)

            # do stuff with sample
            x = self.some_other_model(x)
            return x


Use in production or for inference
----------------------------------
For production or predictions, load weights, freeze the model and use as needed.

Example::

    from pl_bolts.models.autoencoders import VAE

    vae = VAE.load_from_checkpoint(PATH)
    vae.freeze()

    z = ... # z ~ N(0, 1)
    predictions = vae(z)


Train from scratch
------------------
Here's an example on how to train this model from scratch

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    vae = VAE()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(vae)


Customize any part
------------------
To adapt a bolt to research or a new problem, you can change any part of a bolt model.

Ex: Changing priors
^^^^^^^^^^^^^^^^^^^
You might be interested in changing the prior of a VAE

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def init_prior(self, z_mu, z_std):
            P = MyPriorDistribution
            # default is standard normal
            # P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
            return P

        def init_posterior(self, z_mu, z_std):
            Q = MyPosteriorDistribution
            # default is normal(z_mu, z_sigma)
            # Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
            return Q

Ex: Changing encoders
^^^^^^^^^^^^^^^^^^^^^
To change parts of the model (for instance, the encoder or decoder) you could do this

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
            encoder = MyEncoder(...)
            return encoder

        def init_decoder(self, hidden_dim, latent_dim, input_width, input_height):
            decoder = MyDecoder(...)
            return decoder

Ex: Changing optimizer
^^^^^^^^^^^^^^^^^^^^^^
Every bolt is a Lightning module. This means you can modify anything, even the optimizer used.

Example::

    from pl_bolts.models.autoencoders import VAE

    class MyVAE(VAE):

        def configure_optimizers(self):
            return ANOptimizer(...), OrASecondOne(...)

Ex: Custom backward pass
^^^^^^^^^^^^^^^^^^^^^^^^
Again, just a Lightning Module

Example::

    from pl_bolts.models.self_supervised import CPCV2

    class MyCPC(CPCV2):

        def backward(self):
            # do something weird


Command line support
--------------------
Any bolt module can also be trained from the command line

.. code-block:: bash

    cd pl_bolts/models/autoencoders/basic_vae
    python basic_vae_pl_module.py

Each script accepts Argparse arguments for both the lightning trainer and the model

.. code-block:: bash

    python basic_vae_pl_module.py --hidden_dim 128 --latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12

"""
