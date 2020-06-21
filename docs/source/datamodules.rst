.. role:: hidden
    :class: hidden-section

Bolts Datamodules
=================
Class for standardizing the prepare_data, train, validation and test dataloaders for a dataset.

Datamodules have two advantages:

1. You can guarantee that the exact same train, val and test splits can be used across models.
2. You can parameterize your model to be dataset agnostic.

Example::

    from pl_bolts.datamodules import STL10DataLoaders, CIFAR10DataLoaders

    # use the same dataset on different models (with exactly the same splits)
    stl10_model = LitModel(STL10DataLoaders())
    stl10_model = CoolModel(STL10DataLoaders())

    # or make your model dataset agnostic
    cifar10_model = LitModel(CIFAR10DataLoaders())

Build a DataModule
------------------
Use this to build your own consistent train, validation, test splits.

Example::

    from pl_bolts.datamodules import BoltDataLoaders

    class MyDataModule(BoltDataLoaders):

        def __init__(self,...):

        def prepare_data(self):
            # download and do something to your data

        def train_dataloader(self, batch_size):
            return DataLoader(...)

        def val_dataloader(self, batch_size):
            return DataLoader(...)

        def test_dataloader(self, batch_size):
            return DataLoader(...)

Then use this in any model you want.

Example::

    class LitModel(pl.LightningModule):

        def __init__(self, data_module=MyDataModule()):
            super().__init()
            self.dm = data_module

        def prepare_data(self):
            self.dm.prepare_data()

        def train_dataloader(self):
            return self.dm.train_dataloader()

        def val_dataloader(self):
            return self.dm.val_dataloader()

        def test_dataloader(self):
            return self.dm.test_dataloader()


DataLoaders class
^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.bolts_dataloaders_base.BoltDataLoaders
   :noindex: