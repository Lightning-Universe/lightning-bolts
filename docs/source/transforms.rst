.. role:: hidden
    :class: hidden-section

Self-supervised learning Transforms
===================================

These transforms are used in various self-supervised learning approaches.

----------------

CPC transforms
--------------

Transforms used for CPC

CIFAR-10 Train (c)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCTrainTransformsCIFAR10
    :noindex:

CIFAR-10 Eval (c)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCEvalTransformsCIFAR10
    :noindex:

Imagenet Train (c)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCTrainTransformsImageNet128
    :noindex:

Imagenet Eval (c)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCEvalTransformsImageNet128
    :noindex:

STL-10 Train (c)
^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCTrainTransformsSTL10
    :noindex:

STL-10 Eval (c)
^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.cpc.transforms.CPCEvalTransformsSTL10
    :noindex:

-------------------

AMDIM transforms
----------------

Transforms used for AMDIM

CIFAR-10 Train (a)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMTrainTransformsCIFAR10
    :noindex:

CIFAR-10 Eval (a)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMEvalTransformsCIFAR10
    :noindex:

Imagenet Train (a)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMTrainTransformsImageNet128
    :noindex:

Imagenet Eval (a)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMEvalTransformsImageNet128
    :noindex:

STL-10 Train (a)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMTrainTransformsSTL10
    :noindex:

STL-10 Eval (a)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.amdim.transforms.AMDIMEvalTransformsSTL10
    :noindex:

---------------

MOCO V2 transforms
------------------

Transforms used for MOCO V2

CIFAR-10 Train (m2)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2TrainCIFAR10Transforms
    :noindex:

CIFAR-10 Eval (m2)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2EvalCIFAR10Transforms
    :noindex:

Imagenet Train (m2)
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2TrainSTL10Transforms
    :noindex:

Imagenet Eval (m2)
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2EvalSTL10Transforms
    :noindex:

STL-10 Train (m2)
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2TrainImagenetTransforms
    :noindex:

STL-10 Eval (m2)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.moco.transforms.Moco2EvalImagenetTransforms
    :noindex:

---------------

SimCLR transforms
------------------
Transforms used for SimCLR

Train (sc)
^^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.simclr.transforms.SimCLRTrainDataTransform
    :noindex:

Eval (sc)
^^^^^^^^^
.. autoclass:: pl_bolts.models.self_supervised.simclr.transforms.SimCLREvalDataTransform
    :noindex:
