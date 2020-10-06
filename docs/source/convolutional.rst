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


.. autoclass:: pl_bolts.models.vision.segmentation.SemSegment
    :noindex:
