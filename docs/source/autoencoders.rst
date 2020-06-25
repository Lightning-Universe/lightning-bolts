Autoencoder Models
==================
This section houses autoencoders and variational autoencoders.

----------------


Basic AE
^^^^^^^^
This is the simplest autoencoder. You can use it like so

.. code-block:: python

    from pl_bolts.models.autoencoders import AE

    model = AE()
    trainer = Trainer()
    trainer.fit(model)

You can override any part of this AE to build your own variation.

.. code-block:: python

    from pl_bolts.models.autoencoders import AE

    class MyAEFlavor(AE):

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

    from pl_bolts.models.autoencoders import VAE

    model = VAE()
    trainer = Trainer()
    trainer.fit(model)

You can override any part of this VAE to build your own variation.

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def get_posterior(self, mu, std):
            # do something other than the default
            # P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))

            return P

.. autoclass:: pl_bolts.models.autoencoders.VAE
   :noindex: