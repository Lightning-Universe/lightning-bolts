"""
VAE Template
============

This is a basic template for implementing a Variational Autoencoder in PyTorch Lightning.

A default encoder and decoder have been provided but can easily be replaced by custom models.

This template uses the MNIST dataset but image data of any dimension can be fed in as long as the image
 width and image height are even values. For other types of data, such as sound, it will be necessary
 to change the Encoder and Decoder.

The default encoder and decoder are both convolutional with a 128-dimensional hidden layer and
 a 32-dimensional latent space. The model accepts arguments for these dimensions (see example below)
 if you want to use the default encoder + decoder but with different hidden layer and latent layer dimensions.
 The model also assumes a Gaussian prior and a Gaussian approximate posterior distribution.

Use as a feature extractor
--------------------------
For certain projects that require a VAE architecture you could use this as
a module inside the larger system.

>>> from ptl_bolts.models.VAEs import VAE
>>> import pytorch_lightning as pl

>>> class YourResearchModel(pl.LightningModule):
...    def __init__(self):
...        self.vae = VAE.load_from_checkpoint(PATH)
...        self.vae.freeze()
...
...        self.some_other_model = MyModel()
...
...    def forward(self, z):
...        # generate a sample from z ~ N(0,1)
...        x = self.vae(z)
...
...        # do stuff with sample
...        x = self.some_other_model(x)
...        return x


Use in production of for inference
----------------------------------
For production or predictions, load weights, freeze the model and use as needed.

.. code-block:: python

    from pytorch_lightning_bolts.models.VAEs import VAE

    vae = VAE.load_from_checkpoint(PATH)
    vae.freeze()

    z = ... # z ~ N(0, 1)
    predictions = vae(z)


Train from scratch
------------------
Here's an example on how to train this model from scratch

.. code-block:: python

    from pytorch_lightning_bolts.models.VAEs import VAE
    import pytorch_lightning as pl

    vae = VAE()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(vae)


Use for research
----------------
To use the VAE for research, modify any relevant part you need.

For example to change the prior and posterior you could do this

.. code-block:: python

    from pytorch_lightning_bolts.models.VAEs import VAE

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

To change the encoder or decoder you could do this

.. code-block:: python

    from pytorch_lightning_bolts.models.VAEs import VAE

    class MyVAEFlavor(VAE):

        def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
            encoder = MyEncoder(...)
            return encoder

        def init_decoder(self, hidden_dim, latent_dim, input_width, input_height):
            decoder = MyDecoder(...)
            return decoder


Train VAE from the command line
-------------------------------

.. code-block:: bash

    cd pytorch_lightning_bolts/models/vaes/basic_vae
    python vae.py


The vae.py script accepts the following arguments::

    optional arguments:
    --hidden_dim        if using default encoder/decoder - dimension of itermediate (dense) layers before embedding
    --latent_dim        dimension of latent variables z
    --input_width       input image width (must be even) - 28 for MNIST
    --input_height      input image height (must be even) - 28 for MNIST
    --batch_size

    any arguments from pl.Trainer - e.g max_epochs, gpus

For example::

    python vae.py --hidden_dim 128 --latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12

"""
