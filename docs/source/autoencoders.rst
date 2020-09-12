Autoencoders
============
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

You can use the pretrained models present in bolts.

CIFAR-10 pretrained model::

    from pl_bolts.models.autoencoders import AE

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/cifar10/checkpoints/epoch%3D99.ckpt'
    ae = AE.load_from_checkpoint(weight_path, strict=False)

    ae.freeze()

|

- `Tensorboard for CIFAR10 training <https://tensorboard.dev/experiment/W6TmkmFZSEKfRxMkqJTjpA/>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: loss

|

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

You can use the pretrained models present in bolts.

CIFAR-10 pretrained model::

    from pl_bolts.models.autoencoders import VAE

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/cifar10/checkpoints/epoch%3D98.ckpt'
    vae = VAE.load_from_checkpoint(weight_path, strict=False)

    vae.freeze()

|

- `Tensorboard for CIFAR10 training <https://tensorboard.dev/experiment/2OZtEH4yQ2iVSgkxJwM7bw/#scalars>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: reconstruction loss

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: kl

|

STL-10 pretrained model::

    from pl_bolts.models.autoencoders import VAE

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/cifar10/checkpoints/epoch%3D98.ckpt'
    vae = VAE.load_from_checkpoint(weight_path, strict=False)

    vae.freeze()

|

- `Tensorboard for CIFAR10 training <https://tensorboard.dev/experiment/2OZtEH4yQ2iVSgkxJwM7bw/#scalars>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: reconstruction loss

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: kl

|

.. autoclass:: pl_bolts.models.autoencoders.VAE
   :noindex:
