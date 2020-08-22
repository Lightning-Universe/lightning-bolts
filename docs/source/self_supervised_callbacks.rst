.. role:: hidden
    :class: hidden-section

Self-supervised Callbacks
=========================
Useful callbacks for self-supervised learning models

---------------

BYOLMAWeightUpdate
------------------
The exponential moving average weight-update rule from Bring Your Own Latent (BYOL).

.. autoclass:: pl_bolts.callbacks.self_supervised.BYOLMAWeightUpdate
   :noindex:

----------------

SSLOnlineEvaluator
------------------
Appends a MLP for fine-tuning to the given model. Callback has its own mini-inner loop.

.. autoclass:: pl_bolts.callbacks.self_supervised.SSLOnlineEvaluator
   :noindex: