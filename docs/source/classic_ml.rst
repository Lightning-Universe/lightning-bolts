Classic ML Models
=================
This module implements classic machine learning models, which may
or may not be differentiable. The key advantage of this...
These contain PyTorch based classic machine learning models such as linear regression. Unline other
libraries that implement these models, here we use PyTorch to enable multi-GPU, multi-TPU, half-precision training.
These are off-the-shelf autoencoder which can be used for resarch or as feature extractors.
For the non-differentiable models, you can benefit from distributed training

can be plugged into any deep learning pipeline - because we make it work withh differentiable models

---------------

Linear Regression
-----------------


.. code-block:: python

    from pl_bolts.models import LinearRegression
    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataModule
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    model = LinearRegression()
    trainer = pl.Trainer()
    trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())
    trainer.test(test_dataloaders=loaders.test_dataloader())

.. autoclass:: pl_bolts.models.regression.linear_regression.LinearRegression
   :noindex:

