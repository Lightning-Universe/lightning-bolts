GANs
====
Collection of Generative Adversarial Networks

------------

Basic GAN
---------
This is a basic GAN.


.. code-block:: python

    from pl_bolts.models.gans import GAN
    ...
    gan = GAN()
    trainer = Trainer()
    trainer.fit(gan)


.. autoclass:: pl_bolts.models.gans.GAN
   :noindex: