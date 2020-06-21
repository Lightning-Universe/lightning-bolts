Self-supervised
===============
This bolts module houses a collection of all self-supervised learning models.

Self-supervised learning extracts representations of an input by solving a pretext task. In this package,
we implement many of the current state-of-the-art self-supervised algorithms.

Use cases
---------
The models in this module can be used as templates for research, subclassing for similar model research,
or as feature extractors.

For instance, many of the contrastive learning models have been pretrained on many of the torchvision encoders
and can be used as feature extractors.

In this example, we'll load a resnet 18 which was pretrained on imagenet using CPC as the pretext task.

Example::

            from pl_bolts.models.self_supervised import CPCV2

            # load resnet18 pretrained using CPC on imagenet
            model = CPCV2(pretrained='resnet18')
            cpc_resnet18 = model.encoder
            cpc_resnet18.freeze()

            # it supports any torchvision resnet
            model = CPCV2(pretrained='resnet50')

This means you can now extract image representations that were pretrained via unsupervised learning.

Example::

    my_dataset = SomeDataset()
    for batch in my_dataset:
        x, y = batch
        out = cpc_resnet18(x)


-----------------

Contrastive Learning
--------------------
Contrastive self-supervised learning (CSL) is a self-supervised learning approach where we generate representations
of instances such that similar instances are near each other and far from dissimilar ones. This is often done by comparing
triplets of positive, anchor and negative representations.

In this section, we list Lightning implementations of popular contrastive learning approaches.

AMDIM
^^^^^

.. autoclass:: pl_bolts.models.self_supervised.AMDIM
   :noindex:

CPC (V2)
^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.CPCV2
   :noindex:

SimCLR
^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.SimCLR
   :noindex:

Moco (V2)
^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.MocoV2
   :noindex:
