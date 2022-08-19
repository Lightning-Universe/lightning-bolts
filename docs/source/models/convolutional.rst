Convolutional Architectures
===========================
This package lists contributed convolutional architectures.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

--------------


GPT-2
-----

.. autoclass:: pl_bolts.models.vision.GPT2
    :noindex:

-------------

Image GPT
---------

.. autoclass:: pl_bolts.models.vision.ImageGPT
    :noindex:

-------------

Pixel CNN
---------

.. autoclass:: pl_bolts.models.vision.PixelCNN
    :noindex:

-------------

UNet
----

.. autoclass:: pl_bolts.models.vision.UNet
    :noindex:

-------------

Semantic Segmentation
---------------------
Model template to use for semantic segmentation tasks. The model uses a UNet architecture by default. Override any part
of this model to build your own variation.

.. code-block:: python

    from pl_bolts.models.vision import SemSegment
    from pl_bolts.datamodules import KittiDataModule
    import pytorch_lightning as pl

    dm = KittiDataModule('path/to/kitt/dataset/', batch_size=4)
    model = SemSegment(datamodule=dm)
    trainer = pl.Trainer()
    trainer.fit(model)

.. autoclass:: pl_bolts.models.vision.SemSegment
    :noindex:
