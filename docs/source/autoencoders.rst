Autoencoder Models
==================
These are off-the-shelf autoencoder which can be used for resarch or as feature extractors.

Pretrained models
------------------
This is a basic template for implementing a Variational Autoencoder in PyTorch Lightning.

A default encoder and decoder have been provided but can easily be replaced by custom models.

This template uses the MNIST dataset but image data of any dimension can be fed in as long as the image width and image height are even values.
For other types of data, such as sound, it will be necessary to change the Encoder and Decoder.

The default encoder and decoder are both convolutional with a 128-dimensional hidden layer and
a 32-dimensional latent space. The model also assumes a Gaussian prior and a Gaussian approximate posterior distribution.

To use in your project or as a feature extractor:

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    class YourResearchModel(pl.LightningModule):
        def __init__(self):
            self.vae = VAE.load_from_checkpoint(PATH)
            self.vae.freeze()

            self.some_other_model = MyModel()

        def forward(self, z):
            # generate a sample from z ~ N(0,1)
            x = self.vae(z)

            # do stuff with sample
            x = self.some_other_model(x)
            return x

To use in production or for predictions:

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE

    vae = VAE.load_from_checkpoint(PATH)
    vae.freeze()

    z = ... # z ~ N(0, 1)
    predictions = vae(z)

Research use case
-----------------
You can train the VAE on its own:

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    vae = VAE()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(vae)

You can also use as template for research (example of modifying only the prior):

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def get_posterior(self, mu, std):
            # do something other than the default
            # P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))

            return P

Or pass in your own encoders and decoders:

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    encoder = MyEncoder()
    decoder = MyDecoder()

    vae = VAE(encoder=encoder, decoder=decoder)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(vae)

Train the VAE from the command line:

.. code-block:: python

    cd pytorch_lightning_bolts/models/autoencoders/basic_vae
    python vae.py

The vae.py script accepts the following arguments:

.. code-block:: bash

    optional arguments:
    --hidden_dim        if using default encoder/decoder - dimension of itermediate (dense) layers before embedding
    --latent_dim        dimension of latent variables z
    --input_width       input image width (must be even) - 28 for MNIST
    --input_height      input image height (must be even) - 28 for MNIST
    --batch_size

any arguments from pl.Trainer - e.g max_epochs, gpus

.. code-block:: bash

    python vae.py --hidden_dim 128 --latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12

---------------

Autoencoders
------------
The following is a collection of auto-encoders.

Basic AE
^^^^^^^^
This is the simplest autoencoder. You can use it like so

.. code-block:: python

    from pytorch_lightning.models.autoencoders import AE

    model = AE()
    trainer = Trainer()
    trainer.fit(model)

You can override any part of this AE to build your own variation.

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
            encoder = YourSuperFancyEncoder(...)
            return encoder

.. autoclass:: pl_bolts.models.autoencoders.AE
   :noindex:

---------------

Variational Autoencoders
------------------------

Basic VAE
^^^^^^^^^
Use the VAE like so.

.. code-block:: python

    from pytorch_lightning.models.autoencoders import VAE

    model = VAE()
    trainer = Trainer()
    trainer.fit(model)

You can override any part of this VAE to build your own variation.

.. code-block:: python

    from pytorch_lightning_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def get_posterior(self, mu, std):
            # do something other than the default
            # P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))

            return P

.. autoclass:: pl_bolts.models.autoencoders.VAE
   :noindex: