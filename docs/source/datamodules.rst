.. role:: hidden
    :class: hidden-section

Bolts DataModule
================
Datasets in PyTorch, Lightning and general Deep learning research have 4 main parts:

    1. A train split + dataloader
    2. A val split + dataloader
    3. A test split + dataloader
    4. A step to download, split, etc...

Step 4, also needs special care to make sure that it's only done on 1 GPU in a multi-GPU set-up.
In addition, there are other challenges such as models that are built using information from the dataset
such as needing to know image dimensions or number of classes.

A datamodule simplifies all of these parts and integrates seamlessly into Lightning.

.. code-block:: python

    from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule


    datamodule = CIFAR10DataModule()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.train_dataloader()
    test_loader = datamodule.train_dataloader()

And they can be used in lightning modules

.. code-block:: python

    class LitModel(pl.LightningModule):

        def __init__(self, datamodule):
            c, w, h = datamodule.size()
            self.l1 = nn.Linear(128, datamodule.num_classes)
            self.datamodule = datamodule

        def prepare_data(self):
            self.datamodule.prepare_data()

        def train_dataloader(self)
            return self.datamodule.train_dataloader()

        def val_dataloader(self)
            return self.datamodule.val_dataloader()

        def test_dataloader(self)
            return self.datamodule.test_dataloader()

An advantage is that you can parametrize the data of your LightningModule

.. code-block:: python

    model = LitModel(datamodule = CIFAR10DataModule())
    model = LitModel(datamodule = ImagenetDataModule())

Or even bridge between SKLearn or numpy datasets

.. code-block:: python

    from sklearn.datasets import load_boston
    from pl_bolts.datamodules import SklearnDataModule

    X, y = load_boston(return_X_y=True)
    datamodule = SklearnDataModule(X, y)

    model = LitModel(datamodule)


DataModule Advantages
---------------------
Datamodules have two advantages:

    1. You can guarantee that the exact same train, val and test splits can be used across models.
    2. You can parameterize your model to be dataset agnostic.

Example::

    from pl_bolts.datamodules import STL10DataModule, CIFAR10DataModule

    # use the same dataset on different models (with exactly the same splits)
    stl10_model = LitModel(STL10DataModule())
    stl10_model = CoolModel(STL10DataModule())

    # or make your model dataset agnostic
    cifar10_model = LitModel(CIFAR10DataModule())

Build a DataModule
------------------
Use this to build your own consistent train, validation, test splits.

Example::

    from pl_bolts.datamodules import LightningDataModule

    class MyDataModule(LightningDataModule):

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


DataModule class
^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.datamodules.bolts_dataloaders_base.LightningDataModule
   :noindex: