.. role:: hidden
    :class: hidden-section

DataModules
-----------
DataModules (introduced in PyTorch Lightning 0.9.0) decouple the data from a model. A DataModule
is simply a collection of a training dataloder, val dataloader and test dataloader. In addition,
it specifies how to:

- Download/prepare data.
- Train/val/test splits.
- Transform

Then you can use it like this:

Example::

    dm = MNISTDataModule('path/to/data')
    model = LitModel()

    trainer = Trainer()
    trainer.fit(model, datamodule=dm)

Or use it manually with plain PyTorch

Example::

    dm = MNISTDataModule('path/to/data')
    for batch in dm.train_dataloader():
        ...
    for batch in dm.val_dataloader():
        ...
    for batch in dm.test_dataloader():
        ...

Please visit the PyTorch Lightning documentation for more details on DataModules.

- :ref:`vision-datamodules`

  - :ref:`vision-datamodule-supervised-learning`

    - :class:`~pl_bolts.datamodules.binary_emnist_datamodule.BinaryEMNISTDataModule`

    - :class:`~pl_bolts.datamodules.binary_mnist_datamodule.BinaryMNISTDataModule`

    - :class:`~pl_bolts.datamodules.cityscapes_datamodule.CityscapesDataModule`

    - :class:`~pl_bolts.datamodules.cifar10_datamodule.CIFAR10DataModule`

    - :class:`~pl_bolts.datamodules.emnist_datamodule.EMNISTDataModule`

    - :class:`~pl_bolts.datamodules.fashion_mnist_datamodule.FashionMNISTDataModule`

    - :class:`~pl_bolts.datamodules.imagenet_datamodule.ImagenetDataModule`

    - :class:`~pl_bolts.datamodules.mnist_datamodule.MNISTDataModule`

  - :ref:`vision-datamodule-semi-supervised-learning`

    - :class:`~pl_bolts.datamodules.ssl_imagenet_datamodule.SSLImagenetDataModule`

    - :class:`~pl_bolts.datamodules.stl10_datamodule.STL10DataModule`

- Sklearn Datamodule

  - :class:`~pl_bolts.datamodules.sklearn_datamodule.SklearnDataModule`
