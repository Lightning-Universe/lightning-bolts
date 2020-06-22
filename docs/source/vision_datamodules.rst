Vision DataModules
==================
The following are pre-built datamodules.

Supervised learning
--------------------
These are standard vision datasets with the train, test, val splits pre-generated in DataLoaders with
the standard transforms (and Normalization) values

MNIST
^^^^^

.. autoclass:: pl_bolts.datamodules.mnist_dataloaders.MNISTDataModule
    :noindex:

CIFAR-10
^^^^^^^^

.. autoclass:: pl_bolts.datamodules.cifar10_dataloaders.CIFAR10DataModule
    :noindex:

STL-10
^^^^^^

.. autoclass:: pl_bolts.datamodules.stl10_dataloaders.STL10DataLoaders
    :noindex:

Imagenet
^^^^^^^^

.. autoclass:: pl_bolts.datamodules.imagenet_dataloaders.ImagenetDataModule
    :noindex:


Semi-supervised learning
------------------------
The following datasets have support for unlabeled training and semi-supervised learning where only a few examples
are labeled.

Imagenet (ssl)
^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.ssl_imagenet_dataloaders.SSLImagenetDataLoaders
    :noindex: