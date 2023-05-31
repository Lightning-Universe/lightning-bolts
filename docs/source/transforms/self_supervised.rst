.. role:: hidden
    :class: hidden-section

Self-supervised learning
========================

These transforms are used in various self-supervised learning approaches.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

----------------

CPC transforms
--------------

Transforms used for CPC

CIFAR-10 Train (c)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCTrainTransformsCIFAR10
    :noindex:

CIFAR-10 Eval (c)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCEvalTransformsCIFAR10
    :noindex:

Imagenet Train (c)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCTrainTransformsImageNet128
    :noindex:

Imagenet Eval (c)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCEvalTransformsImageNet128
    :noindex:

STL-10 Train (c)
^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCTrainTransformsSTL10
    :noindex:

STL-10 Eval (c)
^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.cpc_transforms.CPCEvalTransformsSTL10
    :noindex:

AMDIM transforms
----------------

Transforms used for AMDIM

CIFAR-10 Train (a)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMTrainTransformsCIFAR10
    :noindex:

CIFAR-10 Eval (a)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMEvalTransformsCIFAR10
    :noindex:

Imagenet Train (a)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMTrainTransformsImageNet128
    :noindex:

Imagenet Eval (a)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMEvalTransformsImageNet128
    :noindex:

STL-10 Train (a)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMTrainTransformsSTL10
    :noindex:

STL-10 Eval (a)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.amdim_transforms.AMDIMEvalTransformsSTL10
    :noindex:

MOCO V2 transforms
------------------

Transforms used for MOCO V2

CIFAR-10 Train (m2)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2TrainCIFAR10Transforms
    :noindex:

CIFAR-10 Eval (m2)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2EvalCIFAR10Transforms
    :noindex:

Imagenet Train (m2)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2TrainSTL10Transforms
    :noindex:

Imagenet Eval (m2)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2EvalSTL10Transforms
    :noindex:

STL-10 Train (m2)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2TrainImagenetTransforms
    :noindex:

STL-10 Eval (m2)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.moco_transforms.MoCo2EvalImagenetTransforms
    :noindex:

SimCLR transforms
------------------
Transforms used for SimCLR

Train (sc)
^^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.simclr_transforms.SimCLRTrainDataTransform
    :noindex:

Eval (sc)
^^^^^^^^^
.. autoclass:: pl_bolts.transforms.self_supervised.simclr_transforms.SimCLREvalDataTransform
    :noindex:


---------------------

Identity class
--------------
Example::

    from pl_bolts.utils import Identity

.. autoclass:: pl_bolts.utils.self_supervised.Identity
    :noindex:

------------

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

--------------

SSL backbone finetuner
----------------------

.. autoclass:: pl_bolts.models.self_supervised.ssl_finetuner.SSLFineTuner
    :noindex:
