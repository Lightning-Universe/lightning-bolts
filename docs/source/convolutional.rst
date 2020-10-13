Convolutional Architectures
===========================
This package lists contributed convolutional architectures.

--------------


GPT-2
-----

.. autoclass:: pl_bolts.models.vision.image_gpt.gpt2.GPT2
    :noindex:

-------------

Image GPT
---------

.. autoclass:: pl_bolts.models.vision.image_gpt.igpt_module.ImageGPT
    :noindex:

-------------

Pixel CNN
---------

.. autoclass:: pl_bolts.models.vision.pixel_cnn.PixelCNN
    :noindex:

-------------

UNet
----

.. autoclass:: pl_bolts.models.vision.unet.UNet
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

.. autoclass:: pl_bolts.models.vision.segmentation.SemSegment
    :noindex:
