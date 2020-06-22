"""
AE Template
============

This is a basic template for implementing an Autoencoder in PyTorch Lightning.

A default encoder and decoder have been provided but can easily be replaced by custom models.

This template uses the MNIST dataset but image data of any dimension can be fed in as long as the image
 width and image height are even values. For other types of data, such as sound, it will be necessary
 to change the Encoder and Decoder.

The default encoder and decoder are both convolutional with a 128-dimensional hidden layer and
 a 32-dimensional latent space. The model accepts arguments for these dimensions (see example below)
 if you want to use the default encoder + decoder but with different hidden layer and latent layer dimensions.
 The model also assumes a Gaussian prior and a Gaussian approximate posterior distribution.

.. code-block:: python

    from pl_bolts.models.autoencoders import AE

    model = AE()
    trainer = pl.Trainer()
    trainer.fit(model)
"""
from pl_bolts.models.autoencoders.basic_ae.components import AEEncoder
