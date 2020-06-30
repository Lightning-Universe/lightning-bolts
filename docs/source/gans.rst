GANs
====
Collection of Generative Adversarial Networks

------------

Basic GAN
---------
This is a vanilla GAN. This model can work on any dataset size but results are shown for MNIST.
Replace the encoder, decoder or any part of the training loop to build a new method.

Implemented by:

    - William Falcon

Example outputs:

    .. image:: _images/gans/basic_gan_interpolate.png
        :width: 400
        :alt: Basic GAN generated samples

Loss curves:

    .. image:: _images/gans/basic_gan_dloss.png
        :width: 200
        :alt: Basic GAN disc loss

    .. image:: _images/gans/basic_gan_gloss.png
        :width: 200
        :alt: Basic GAN gen loss

.. code-block:: python

    from pl_bolts.models.gans import GAN
    ...
    gan = GAN()
    trainer = Trainer()
    trainer.fit(gan)


.. autoclass:: pl_bolts.models.gans.GAN
   :noindex: