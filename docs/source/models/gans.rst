GANs
====
Collection of Generative Adversarial Networks

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

------------

Basic GAN
---------
This is a vanilla GAN. This model can work on any dataset size but results are shown for MNIST.
Replace the encoder, decoder or any part of the training loop to build a new method, or simply
finetune on your data.

Implemented by:

    - William Falcon

Example outputs:

    .. image:: ../_images/gans/basic_gan_interpolate.jpg
        :width: 400
        :alt: Basic GAN generated samples

Loss curves:

    .. image:: ../_images/gans/basic_gan_dloss.jpg
        :width: 200
        :alt: Basic GAN disc loss

    .. image:: ../_images/gans/basic_gan_gloss.jpg
        :width: 200
        :alt: Basic GAN gen loss

.. code-block:: python

    from pl_bolts.models.gans import GAN
    ...
    gan = GAN()
    trainer = Trainer()
    trainer.fit(gan)


.. autoclass:: pl_bolts.models.gans.GAN
   :noindex:

DCGAN
---------
DCGAN implementation from the paper `Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks <https://arxiv.org/pdf/1511.06434.pdf>`_. The implementation is based on the version from
PyTorch's `examples <https://github.com/pytorch/examples/blob/master/dcgan/main.py>`_.

Implemented by:

    - `Christoph Clement <https://github.com/chris-clem>`_

Example MNIST outputs:

    .. image:: ../_images/gans/dcgan_mnist_outputs.png
        :width: 400
        :alt: DCGAN generated MNIST samples

Example LSUN bedroom outputs:

    .. image:: ../_images/gans/dcgan_lsun_outputs.png
        :width: 400
        :alt: DCGAN generated LSUN bedroom samples

MNIST Loss curves:

    .. image:: ../_images/gans/dcgan_mnist_dloss.png
        :width: 200
        :alt: DCGAN MNIST disc loss

    .. image:: ../_images/gans/dcgan_mnist_gloss.png
        :width: 200
        :alt: DCGAN MNIST gen loss

LSUN Loss curves:

    .. image:: ../_images/gans/dcgan_lsun_dloss.png
        :width: 200
        :alt: DCGAN LSUN disc loss

    .. image:: ../_images/gans/dcgan_lsun_gloss.png
        :width: 200
        :alt: DCGAN LSUN gen loss

.. autoclass:: pl_bolts.models.gans.DCGAN
   :noindex:


SRGAN
---------
SRGAN implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network <https://arxiv.org/pdf/1609.04802.pdf>`_. The implementation is based on the version from
`deeplearning.ai <https://github.com/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb>`_.

Implemented by:

    - `Christoph Clement <https://github.com/chris-clem>`_

MNIST results:

    SRGAN MNIST with scale factor of 2 (left: low res, middle: generated high res, right: ground truth high res):

        .. image:: ../_images/gans/srgan-mnist-scale_factor=2.png
            :width: 200
            :alt: SRGAN MNIST with scale factor of 2

    SRGAN MNIST with scale factor of 4:

        .. image:: ../_images/gans/srgan-mnist-scale_factor=4.png
            :width: 200
            :alt: SRGAN MNIST with scale factor of 4

    SRResNet pretraining command used::
        >>>  python srresnet_module.py --dataset=mnist --data_dir=~/Data --scale_factor=4 --save_model_checkpoint \
        --batch_size=16 --num_workers=2 --gpus=4 --accelerator=ddp --precision=16 --max_steps=25000

    SRGAN training command used::
        >>>  python srgan_module.py --dataset=mnist --data_dir=~/Data --scale_factor=4 --batch_size=16 \
        --num_workers=2 --scheduler_step=29 --gpus=4 --accelerator=ddp --precision=16 --max_steps=50000

STL10 results:

    SRGAN STL10 with scale factor of 2:

        .. image:: ../_images/gans/srgan-stl10-scale_factor=2.png
            :width: 200
            :alt: SRGAN STL10 with scale factor of 2

    SRGAN STL10 with scale factor of 4:

        .. image:: ../_images/gans/srgan-stl10-scale_factor=4.png
            :width: 200
            :alt: SRGAN STL10 with scale factor of 4

    SRResNet pretraining command used::
        >>>  python srresnet_module.py --dataset=stl10 --data_dir=~/Data --scale_factor=4 --save_model_checkpoint \
        --batch_size=16 --num_workers=2 --gpus=4 --accelerator=ddp --precision=16 --max_steps=25000

    SRGAN training command used::
        >>>  python srgan_module.py --dataset=stl10 --data_dir=~/Data --scale_factor=4 --batch_size=16 \
        --num_workers=2 --scheduler_step=29 --gpus=4 --accelerator=ddp --precision=16 --max_steps=50000

CelebA results:

    SRGAN CelebA with scale factor of 2:

        .. image:: ../_images/gans/srgan-celeba-scale_factor=2.png
            :width: 200
            :alt: SRGAN CelebA with scale factor of 2

    SRGAN CelebA with scale factor of 4:

        .. image:: ../_images/gans/srgan-celeba-scale_factor=4.png
            :width: 200
            :alt: SRGAN CelebA with scale factor of 4

    SRResNet pretraining command used::
        >>>  python srresnet_module.py --dataset=celeba --data_dir=~/Data --scale_factor=4 --save_model_checkpoint \
        --batch_size=16 --num_workers=2 --gpus=4 --accelerator=ddp --precision=16 --max_steps=25000

    SRGAN training command used::
        >>>  python srgan_module.py --dataset=celeba --data_dir=~/Data --scale_factor=4 --batch_size=16 \
        --num_workers=2 --scheduler_step=29 --gpus=4 --accelerator=ddp --precision=16 --max_steps=50000

.. autoclass:: pl_bolts.models.gans.SRGAN
   :noindex:

.. autoclass:: pl_bolts.models.gans.SRResNet
   :noindex:
