GANs
====
Stuff about gans

Basic GAN
---------
This is a basic GAN.


Example:

    >>> from pytorch_lightning.models.gans import GAN
    ...
    >>> gan = GAN()

Train::

    trainer = Trainer()
    trainer.fit(gan)


.. autoclass:: pl_bolts.models.gans.GAN
   :noindex: