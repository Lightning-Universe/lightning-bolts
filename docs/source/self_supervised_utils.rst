Self-supervised learning
========================
Collection of useful functions for self-supervised learning

---------------------

Identity class
--------------
Example::

    from pl_bolts.utils import Identity

.. autoclass:: pl_bolts.utils.self_supervised.Identity
    :noindex:

SSL-ready resnets
--------------------
Torchvision resnets with the fc layers removed and with the ability to return all feature maps instead of just the
last one.

Example::

    from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

    resnet = torchvision_ssl_encoder('resnet18', pretrained=False, return_all_feature_maps=True)
    x = torch.rand(3, 3, 32, 32)

    feat_maps = resnet(x)

.. autofunction:: pl_bolts.utils.self_supervised.torchvision_ssl_encoder
    :noindex:
