GANs
====
Collection of Generative Adversarial Networks

------------

Basic GAN
---------
This is a vanilla GAN. This model can work on any dataset size but results are shown for MNIST.
Replace the encoder, decoder or any part of the training loop to build a new method, or simply
finetune on your data.

Implemented by:

    - William Falcon

Example outputs:

    .. image:: _images/gans/basic_gan_interpolate.jpg
        :width: 400
        :alt: Basic GAN generated samples

Loss curves:

    .. image:: _images/gans/basic_gan_dloss.jpg
        :width: 200
        :alt: Basic GAN disc loss

    .. image:: _images/gans/basic_gan_gloss.jpg
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

DCGAN
---------
DCGAN implementation from the paper `Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks <https://arxiv.org/pdf/1511.06434.pdf>`_. The implementation is based on the version from
PyTorch's `examples <https://github.com/pytorch/examples/blob/master/dcgan/main.py>`_.

Implemented by:

    - `Christoph Clement <https://github.com/chris-clem>`_

Example outputs:

    .. image:: _images/gans/dcgan_outputs.png
        :width: 400
        :alt: DCGAN generated samples

Loss curves:

    .. image:: _images/gans/dcgan_dloss.png
        :width: 200
        :alt: DCGAN disc loss

    .. image:: _images/gans/dcgan_gloss.png
        :width: 200
        :alt: DCGAN gen loss