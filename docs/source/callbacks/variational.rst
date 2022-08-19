.. role:: hidden
    :class: hidden-section

Variational Callbacks
=====================
Useful callbacks for GANs, variational-autoencoders or anything with latent spaces.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

---------------

Latent Dim Interpolator
-----------------------
Interpolates latent dims.

Example output:

    .. image:: ../_images/gans/basic_gan_interpolate.jpg
        :width: 400
        :alt: Example latent space interpolation

.. autoclass:: pl_bolts.callbacks.variational.LatentDimInterpolator
   :noindex:
