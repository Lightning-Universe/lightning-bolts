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
    # download data and setup dataloaders
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        ...
    for batch in dm.val_dataloader():
        ...
    for batch in dm.test_dataloader():
        ...

Please visit the PyTorch Lightning documentation for more details on DataModules
