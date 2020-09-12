"""
VAE Template
============

This is a basic template for implementing a Variational Autoencoder in PyTorch Lightning.

A default encoder and decoder have been provided but can easily be replaced by custom models.

This template uses the CIFAR10 dataset but image data of any dimension can be fed in as long as the image
 width and image height are even values. For other types of data, such as sound, it will be necessary
 to change the Encoder and Decoder.

The default encoder is a resnet18 backbone followed by linear layers which map representations
 to mu and var. The default decoder mirrors the encoder architecture and is similar to an inverted
 resnet18. The model also assumes a Gaussian prior and a Gaussian approximate posterior distribution.
"""
