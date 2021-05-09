.. role:: hidden
    :class: hidden-section

Vision Callbacks
================
Useful callbacks for vision models

---------------

Confused Logit
--------------
Shows how the input would have to change to move the prediction from one logit to the other


Example outputs:

    .. image:: _images/vision/confused_logit.png
        :width: 400
        :alt: Example of prediction confused between 5 and 8

.. autoclass:: pl_bolts.callbacks.vision.confused_logit.ConfusedLogitCallback
   :noindex:


---------------

Tensorboard Image Generator
---------------------------
Generates images from a generative model and plots to tensorboard

.. autoclass:: pl_bolts.callbacks.vision.image_generation.TensorboardGenerativeModelImageSampler
   :noindex:
