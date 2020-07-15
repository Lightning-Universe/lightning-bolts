.. role:: hidden
    :class: hidden-section

Variational Callbacks
=====================
Useful callbacks for GANs, variational-autoencoders or anything with latent spaces.

---------------

Latent Dim Interpolator
-----------------------
Interpolates latent dims.

Example outputs:

    .. image:: _images/gans/basic_gan_interpolate.jpg
        :width: 400
        :alt: Example of prediction confused between 5 and 8

.. autoclass:: pl_bolts.callbacks.variational.LatentDimInterpolator
   :noindex:
