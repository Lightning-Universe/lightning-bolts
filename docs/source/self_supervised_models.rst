Self-supervised
===============
This bolts module houses a collection of all self-supervised learning models.

Self-supervised learning extracts representations of an input by solving a pretext task. In this package,
we implement many of the current state-of-the-art self-supervised algorithms.

Self-supervised models are trained with unlabeled datasets

--------------

Use cases
---------
Here are some use cases for the self-supervised package.

Extracting image features
^^^^^^^^^^^^^^^^^^^^^^^^^
The models in this module are trained unsupervised and thus can capture better image representations (features).

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

----------------

Train with unlabeled data
^^^^^^^^^^^^^^^^^^^^^^^^^
These models are perfect for training from scratch when you have a huge set of unlabeled images

.. code-block:: python

    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform


    train_dataset = MyDataset(transforms=SimCLRTrainDataTransform())
    val_dataset = MyDataset(transforms=SimCLREvalDataTransform())

    # simclr needs a lot of compute!
    model = SimCLR()
    trainer = Trainer(tpu_cores=128)
    trainer.fit(
        model,
        DataLoader(train_dataset),
        DataLoader(val_dataset),
    )

Research
^^^^^^^^
Mix and match any part, or subclass to create your own new method

.. code-block:: python

    from pl_bolts.models.self_supervised import CPCV2
    from pl_bolts.losses.self_supervised_learning import FeatureMapContrastiveTask

    amdim_task = FeatureMapContrastiveTask(comparisons='01, 11, 02', bidirectional=True)
    model = CPCV2(contrastive_task=amdim_task)

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

Moco (V2)
^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.MocoV2
   :noindex:

SimCLR
^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.SimCLR
   :noindex:
