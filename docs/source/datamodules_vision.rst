Vision DataModules
==================
The following are pre-built datamodules for computer-vision.

-------------

Supervised learning
-------------------
These are standard vision datasets with the train, test, val splits pre-generated in DataLoaders with
the standard transforms (and Normalization) values

BinaryEMNIST
^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.binary_emnist_datamodule.BinaryEMNISTDataModule

BinaryMNIST
^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.binary_mnist_datamodule.BinaryMNISTDataModule

CityScapes
^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.cityscapes_datamodule.CityscapesDataModule

CIFAR-10
^^^^^^^^

.. autoclass:: pl_bolts.datamodules.cifar10_datamodule.CIFAR10DataModule

EMNIST
^^^^^^

.. autoclass:: pl_bolts.datamodules.emnist_datamodule.EMNISTDataModule

FashionMNIST
^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.fashion_mnist_datamodule.FashionMNISTDataModule

Imagenet
^^^^^^^^

.. autoclass:: pl_bolts.datamodules.imagenet_datamodule.ImagenetDataModule

MNIST
^^^^^

.. autoclass:: pl_bolts.datamodules.mnist_datamodule.MNISTDataModule

Semi-supervised learning
------------------------
The following datasets have support for unlabeled training and semi-supervised learning where only a few examples
are labeled.

Imagenet (ssl)
^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.ssl_imagenet_datamodule.SSLImagenetDataModule

STL-10
^^^^^^

.. autoclass:: pl_bolts.datamodules.stl10_datamodule.STL10DataModule
