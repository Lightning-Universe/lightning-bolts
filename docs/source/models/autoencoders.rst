Autoencoders
============
This section houses autoencoders and variational autoencoders.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

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

.. testcode::

    from pl_bolts.models.autoencoders import AE

    class MyAEFlavor(AE):

        def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
            encoder = YourSuperFancyEncoder(...)
            return encoder

You can use the pretrained models present in bolts.

CIFAR-10 pretrained model:

.. code-block:: python

    from pl_bolts.models.autoencoders import AE

    ae = AE(input_height=32)
    print(AE.pretrained_weights_available())
    ae = ae.from_pretrained('cifar10-resnet18')

    ae.freeze()

|

- `Tensorboard for AE on CIFAR-10 <https://tensorboard.dev/experiment/3p86iyA9TGaDo6AYtpfzrw/>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/ae-val-loss.png
    :width: 400
    :alt: loss

|

Reconstructions:

Both input and generated images are normalized versions as the training was done with such images.

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/ae_input.png
    :width: 400
    :alt: input

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/ae_output.png
    :width: 400
    :alt: recon

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

CIFAR-10 pretrained model:

.. code-block::

    from pl_bolts.models.autoencoders import VAE

    vae = VAE(input_height=32)
    print(VAE.pretrained_weights_available())
    vae = vae.from_pretrained('cifar10-resnet18')

    vae.freeze()

|

- `Tensorboard for VAE on CIFAR-10 <https://tensorboard.dev/experiment/lIrlQ8uMSwSeM9MAf3Wxig/>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-cifar10/vae-val-recon-cifar10.png
    :width: 400
    :alt: reconstruction loss

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-cifar10/vae-val-kl-cifar10.png
    :width: 400
    :alt: kl

|

Reconstructions:

Both input and generated images are normalized versions as the training was done with such images.

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-cifar10/vae_input.png
    :width: 400
    :alt: input

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-cifar10/vae_output.png
    :width: 400
    :alt: recon

|

STL-10 pretrained model::

    from pl_bolts.models.autoencoders import VAE

    vae = VAE(input_height=96, first_conv=True)
    print(VAE.pretrained_weights_available())
    vae = vae.from_pretrained('cifar10-resnet18')

    vae.freeze()

|

- `Tensorboard for VAE on STL-10 <https://tensorboard.dev/experiment/Kg3P3c2xTjyRe082gGpglg/>`_

Training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-stl10/vae-val-recon-stl10.png
    :width: 400
    :alt: reconstruction loss

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-stl10/vae-val-kl-stl10.png
    :width: 400
    :alt: kl

|

.. autoclass:: pl_bolts.models.autoencoders.VAE
   :noindex:
