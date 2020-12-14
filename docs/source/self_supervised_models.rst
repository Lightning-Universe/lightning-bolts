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
    :width: 400
    :alt: pretraining validation loss

|

Fine-tuning:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/online-finetuning-cpc-cifar10.png
    :width: 400
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
    :width: 400
    :alt: pretraining validation loss

|

Fine-tuning:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/online-finetuning-cpc-stl10.png
    :width: 400
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
    - `Ananya Harsh Jha <https://github.com/ananyahjha93>`_

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
    model = SimCLR(num_samples=dm.num_samples, batch_size=dm.batch_size, dataset='cifar10')

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
     - `~94.00 <https://github.com/google-research/simclr#finetuning-the-linear-head-linear-eval>`_
     - resnet50
     - LARS
     - 2048
     - 800
     - TPUs
     - 1.0/1.5
   * - Ours
     - 88.50
     - `resnet50 <https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/resnets.py#L301-L309>`_
     - `LARS-SGD <https://pytorch-lightning-bolts.readthedocs.io/en/latest/api/pl_bolts.optimizers.lars_scheduling.html#pl_bolts.optimizers.lars_scheduling.LARSWrapper>`_
     - 2048
     - 800 (4 hours)
     - 8 V100 (16GB)
     - 1.5

|

CIFAR-10 pretrained model::

    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    simclr.freeze()

|

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/val_loss.png
    :width: 400
    :alt: pretraining validation loss

|

Fine-tuning (Single layer MLP, 1024 hidden units):

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/val_acc.png
    :width: 400
    :alt: finetuning validation accuracy

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp2_acc_867/test_acc.png
    :width: 400
    :alt: finetuning test accuracy

|

To reproduce::

    # pretrain
    python simclr_module.py
        --gpus 8
        --dataset cifar10
        --batch_size 256
        -- num_workers 16
        --optimizer sgd
        --learning_rate 1.5
        --lars_wrapper
        --exclude_bn_bias
        --max_epochs 800
        --online_ft

    # finetune
    python simclr_finetuner.py
        --gpus 4
        --ckpt_path path/to/simclr/ckpt
        --dataset cifar10
        --batch_size 64
        --num_workers 8
        --learning_rate 0.3
        --num_epochs 100

Imagenet baseline
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
     - `~69.3 <https://github.com/google-research/simclr#finetuning-the-linear-head-linear-eval>`_
     - resnet50
     - LARS
     - 4096
     - 800
     - TPUs
     - 4.8
   * - Ours
     - 68.4
     - `resnet50 <https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/resnets.py#L301-L309>`_
     - `LARS-SGD <https://pytorch-lightning-bolts.readthedocs.io/en/latest/api/pl_bolts.optimizers.lars_scheduling.html#pl_bolts.optimizers.lars_scheduling.LARSWrapper>`_
     - 4096
     - 800
     - 64 V100 (16GB)
     - 4.8

|

Imagenet pretrained model::

    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    simclr.freeze()

|

To reproduce::

    # pretrain
    python simclr_module.py
        --dataset imagenet
        --data_path path/to/imagenet

    # finetune
    python simclr_finetuner.py
        --gpus 8
        --ckpt_path path/to/simclr/ckpt
        --dataset imagenet
        --data_dir path/to/imagenet/dataset
        --batch_size 256
        --num_workers 16
        --learning_rate 0.8
        --nesterov True
        --num_epochs 90

SimCLR API
**********

.. autoclass:: pl_bolts.models.self_supervised.SimCLR
   :noindex:

---------

SwAV
^^^^

PyTorch Lightning implementation of `SwAV <https://arxiv.org/abs/2006.09882>`_
Adapted from the `official implementation <https://github.com/facebookresearch/swav>`_

Paper authors: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin.

Implementation adapted by:

    - `Ananya Harsh Jha <https://github.com/ananyahjha93>`_

To Train::

    import pytorch_lightning as pl
    from pl_bolts.models.self_supervised import SwAV
    from pl_bolts.datamodules import STL10DataModule
    from pl_bolts.models.self_supervised.swav.transforms import (
        SwAVTrainDataTransform, SwAVEvalDataTransform
    )
    from pl_bolts.transforms.dataset_normalizations import stl10_normalization

    # data
    batch_size = 128
    dm = STL10DataModule(data_dir='.', batch_size=batch_size)
    dm.train_dataloader = dm.train_dataloader_mixed
    dm.val_dataloader = dm.val_dataloader_mixed

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=stl10_normalization()
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=stl10_normalization()
    )

    # model
    model = SwAV(
        gpus=1,
        num_samples=dm.num_unlabeled_samples,
        dataset='stl10',
        batch_size=batch_size
    )

    # fit
    trainer = pl.Trainer(precision=16)
    trainer.fit(model)

Pre-trained ImageNet
*****************

We have included an option to directly load
`ImageNet weights <https://github.com/facebookresearch/swav>`_ provided by FAIR into bolts.

You can load the pretrained model using:

ImageNet pretrained model::

    from pl_bolts.models.self_supervised import SwAV

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
    swav = SwAV.load_from_checkpoint(weight_path, strict=True)

    swav.freeze()

|

STL-10 baseline
*****************

The original paper does not provide baselines on STL10.

.. list-table:: STL-10 implementation results
   :widths: 18 15 25 15 10 20 20 20 10
   :header-rows: 1

   * - Implementation
     - test acc
     - Encoder
     - Optimizer
     - Batch
     - Queue used
     - Epochs
     - Hardware
     - LR
   * - Ours
     - `86.72 <https://tensorboard.dev/experiment/w2pq3bPPSxC4VIm5udhA2g/>`_
     - SwAV resnet50
     - `LARS <https://pytorch-lightning-bolts.readthedocs.io/en/latest/api/pl_bolts.optimizers.lars_scheduling.html#pl_bolts.optimizers.lars_scheduling.LARSWrapper>`_
     - 128
     - No
     - 100 (~9 hr)
     - 1 V100 (16GB)
     - 1e-3

|

- `Pre-training tensorboard link <https://tensorboard.dev/experiment/68jet8o4RdK34u5kUXLedg/>`_

STL-10 pretrained model::

    from pl_bolts.models.self_supervised import SwAV

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/checkpoints/swav_stl10.pth.tar'
    swav = SwAV.load_from_checkpoint(weight_path, strict=False)

    swav.freeze()

|

Pre-training:

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/pretraining-val-loss.png
    :width: 400
    :alt: pretraining validation loss

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/online-finetuning-val-acc.png
    :width: 400
    :alt: online finetuning validation acc

|

Fine-tuning (Single layer MLP, 1024 hidden units):

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/fine-tune-val-acc.png
    :width: 400
    :alt: finetuning validation accuracy

.. figure:: https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/fine-tune-val-loss.png
    :width: 400
    :alt: finetuning validation loss

|

To reproduce::

    # pretrain
    python swav_module.py
        --online_ft
        --gpus 1
        --lars_wrapper
        --batch_size 128
        --learning_rate 1e-3
        --gaussian_blur
        --queue_length 0
        --jitter_strength 1.
        --nmb_prototypes 512

    # finetune
    python swav_finetuner.py
    --gpus 8
    --ckpt_path path/to/simclr/ckpt
    --dataset imagenet
    --data_dir path/to/imagenet/dataset
    --batch_size 256
    --num_workers 16
    --learning_rate 0.8
    --nesterov True
    --num_epochs 90

Imagenet baseline
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
   * - Original
     - 75.3
     - resnet50
     - LARS
     - 4096
     - 800
     - 64 V100s
     - 4.8
   * - Ours
     - 74
     - `resnet50 <https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/resnets.py#L301-L309>`_
     - `LARS-SGD <https://pytorch-lightning-bolts.readthedocs.io/en/latest/api/pl_bolts.optimizers.lars_scheduling.html#pl_bolts.optimizers.lars_scheduling.LARSWrapper>`_
     - 4096
     - 800
     - 64 V100 (16GB)
     - 4.8

|

Imagenet pretrained model::

    from pl_bolts.models.self_supervised import SwAV

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt'
    swav = SwAV.load_from_checkpoint(weight_path, strict=False)

    swav.freeze()

|

SwAV API
********

.. autoclass:: pl_bolts.models.self_supervised.SwAV
   :noindex:
