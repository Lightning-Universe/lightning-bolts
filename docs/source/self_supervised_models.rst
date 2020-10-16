Self-supervised Learning
========================
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

Contrastive Learning Models
---------------------------
Contrastive self-supervised learning (CSL) is a self-supervised learning approach where we generate representations
of instances such that similar instances are near each other and far from dissimilar ones. This is often done by comparing
triplets of positive, anchor and negative representations.

In this section, we list Lightning implementations of popular contrastive learning approaches.

AMDIM
^^^^^

.. autoclass:: pl_bolts.models.self_supervised.AMDIM
   :noindex:

---------

BYOL
^^^^

.. autoclass:: pl_bolts.models.self_supervised.BYOL
   :noindex:

---------

CPC (V2)
^^^^^^^^

PyTorch Lightning implementation of `Data-Efficient Image Recognition with Contrastive
Predictive Coding <https://arxiv.org/abs/1905.09272>`_

Paper authors: (Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi,
Carl Doersch, S. M. Ali Eslami, Aaron van den Oord).

Model implemented by:

    - `William Falcon <https://github.com/williamFalcon>`_
    - `Tullie Murrell <https://github.com/tullie>`_

To Train::

    import pytorch_lightning as pl
    from pl_bolts.models.self_supervised import CPCV2
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.models.self_supervised.cpc import (
        CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10)

    # data
    dm = CIFAR10DataModule(num_workers=0)
    dm.train_transforms = CPCTrainTransformsCIFAR10()
    dm.val_transforms = CPCEvalTransformsCIFAR10()

    # model
    model = CPCV2()

    # fit
    trainer = pl.Trainer()
    trainer.fit(model, dm)

To finetune::

    python cpc_finetuner.py
        --ckpt_path path/to/checkpoint.ckpt
        --dataset cifar10
        --gpus 1

CIFAR-10 and STL-10 baselines
*****************************

CPCv2 does not report baselines on CIFAR-10 and STL-10 datasets.
Results in table are reported from the
`YADIM <https://arxiv.org/pdf/2009.00104.pdf>`_ paper.

.. list-table:: CPCv2 implementation results
   :widths: 18 15 25 15 10 20 20 10
   :header-rows: 1

   * - Dataset
     - test acc
     - Encoder
     - Optimizer
     - Batch
     - Epochs
     - Hardware
     - LR
   * - CIFAR-10
     - 84.52
     - `CPCresnet101 <https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/cpc/networks.py#L103>`_
     - Adam
     - 64
     - 1000 (upto 24 hours)
     - 1 V100 (32GB)
     - 4e-5
   * - STL-10
     - 78.36
     - `CPCresnet101 <https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/cpc/networks.py#L103>`_
     - Adam
     - 144
     - 1000 (upto 72 hours)
     - 4 V100 (32GB)
     - 1e-4
   * - ImageNet
     - 54.82
     - `CPCresnet101 <https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/cpc/networks.py#L103>`_
     - Adam
     - 3072
     - 1000 (upto 21 days)
     - 64 V100 (32GB)
     - 4e-5

|

CIFAR-10 pretrained model::

    from pl_bolts.models.self_supervised import CPCV2

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/epoch%3D474.ckpt'
    cpc_v2 = CPCV2.load_from_checkpoint(weight_path, strict=False)

    cpc_v2.freeze()

|

- `Tensorboard for CIFAR10 <https://tensorboard.dev/experiment/8m1aX0gcQ7aEmH0J7kbBtg/#scalars>`_

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/cpc-cifar10-val.png
    :width: 200
    :alt: pretraining validation loss

|

Fine-tuning:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/online-finetuning-cpc-cifar10.png
    :width: 200
    :alt: online finetuning accuracy

|

STL-10 pretrained model::

    from pl_bolts.models.self_supervised import CPCV2

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/epoch%3D624.ckpt'
    cpc_v2 = CPCV2.load_from_checkpoint(weight_path, strict=False)

    cpc_v2.freeze()

|

- `Tensorboard for STL10 <https://tensorboard.dev/experiment/hgYOq0TVQfOwGHLjiBVggA/#scalars>`_

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/cpc-stl10-val.png
    :width: 200
    :alt: pretraining validation loss

|

Fine-tuning:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/online-finetuning-cpc-stl10.png
    :width: 200
    :alt: online finetuning accuracy

|

ImageNet pretrained model::

    from pl_bolts.models.self_supervised import CPCV2

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/checkpoints/epoch%3D526.ckpt'
    cpc_v2 = CPCV2.load_from_checkpoint(weight_path, strict=False)

    cpc_v2.freeze()

|

- `Tensorboard for ImageNet <https://tensorboard.dev/experiment/7li8AqcnQdigDA33LzfDMA/#scalars>`_

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/cpc-imagenet-val.png
    :width: 200
    :alt: pretraining validation loss

|

Fine-tuning:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/online-finetuning-cpc-imagenet.png
    :width: 200
    :alt: online finetuning accuracy

|

CPCV2 API
*********

.. autoclass:: pl_bolts.models.self_supervised.CPCV2
   :noindex:

---------

Moco (V2)
^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.MocoV2
   :noindex:

---------

SimCLR
^^^^^^

PyTorch Lightning implementation of `SimCLR <https://arxiv.org/abs/2006.10029>`_

Paper authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.

Model implemented by:

    - `William Falcon <https://github.com/williamFalcon>`_
    - `Tullie Murrell <https://github.com/tullie>`_

To Train::

    import pytorch_lightning as pl
    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.models.self_supervised.simclr.transforms import (
        SimCLREvalDataTransform, SimCLRTrainDataTransform)

    # data
    dm = CIFAR10DataModule(num_workers=0)
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)

    # model
    model = SimCLR(num_samples=dm.num_samples, batch_size=dm.batch_size)

    # fit
    trainer = pl.Trainer()
    trainer.fit(model, dm)

CIFAR-10 baseline
*****************
.. list-table:: Cifar-10 implementation results
   :widths: 18 15 25 15 10 20 20 10
   :header-rows: 1

   * - Implementation
     - test acc
     - Encoder
     - Optimizer
     - Batch
     - Epochs
     - Hardware
     - LR
   * - `Original <https://github.com/google-research/simclr#finetuning-the-linear-head-linear-eval>`_
     - `92.00? <https://github.com/google-research/simclr#finetuning-the-linear-head-linear-eval>`_
     - resnet50
     - LARS
     - 512
     - 1000
     - 1 V100 (32GB)
     - 1.0
   * - Ours
     - `85.68 <https://tensorboard.dev/experiment/GlS1eLXMQsqh3T5DAec6UQ/#scalars>`_
     - `resnet50 <https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/resnets.py#L301-L309>`_
     - `LARS <https://pytorch-lightning-bolts.readthedocs.io/en/latest/api/pl_bolts.optimizers.lars_scheduling.html#pl_bolts.optimizers.lars_scheduling.LARSWrapper>`_
     - 512
     - 960 (12 hr)
     - 1 V100 (32GB)
     - 1e-6

|

CIFAR-10 pretrained model::

    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    simclr.freeze()

|

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/val_loss.png
    :width: 200
    :alt: pretraining validation loss

|

Fine-tuning (Single layer MLP, 1024 hidden units):

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/val_acc.png
    :width: 200
    :alt: finetuning validation accuracy

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/test_acc.png
    :width: 200
    :alt: finetuning test accuracy

|

To reproduce::

    # pretrain
    python simclr_module.py
        --gpus 1
        --dataset cifar10
        --batch_size 512
        --learning_rate 1e-06
        --num_workers 8

    # finetune
    python simclr_finetuner.py
        --ckpt_path path/to/epoch=xyz.ckpt
        --gpus 1

SimCLR API
**********

.. autoclass:: pl_bolts.models.self_supervised.SimCLR
   :noindex:
