Introduction Guide
==================
Welcome to bolts - The community collection of Lightning models, callbacks and modules which can be "bolted" onto
PyTorch projects.

Contributed bolts that you add are:

    1. Tested
    2. Standardized
    3. Documented
    4. Well maintained

-------------

What is it?
-----------
Lightning Bolts is a collection of Models, Callbacks and other goodies implemented in PyTorch Lightning.

Bolts models are designed to bootstrap research or to be used in production. Here are ways in which bolts can be used

1. As a feature extractor for production and research systems.

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (pretrained on Imagenet)
    pretrained_model = VAE(pretrained='imagenet')
    pretrained_model.freeze()

2. Subclass and override to try new research ideas.

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAE(VAE):

        def elbo_loss(self, x, P, Q):
            # maybe your research is about a new type of loss

.. code-block:: python

    from pl_bolts.models.gans import GAN

    class MyGAN(VAE):

        def init_generator(self, img_dim):
            # do your own generator
            generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_dim)
            return generator

        def generator_step(self, x):
            # maybe try a different generator step?


3. Fully pre-built models that can be fine-tuned on your data.

.. code-block:: python

    from pl_bolts.models.self_supervised import CPCV2

    # feature extractor (pretrained on Imagenet)
    cpc_model = CPCV2(encoder='resnet18', pretrained='imagenet')
    resnet18 = cpc_model.encoder
    resnet18.freeze()

4. Fully contained algorithms that can be trained on your data.

.. code-block:: python

    from pl_bolts.models.self_supervised import SimCLR

    model = SimCLR()
    train_data = DataLoader(yourData())
    val_data = DataLoader(yourData())

    trainer = Trainer()
    trainer.fit(model, train_data, val_data)

5. Fully compatible with GPUs, TPUs, 16-bit precision, etc because of PyTorch Lightning.

.. code-block:: python

    model = SimCLR()

    trainer = Trainer(num_nodes=8, gpus=8)
    trainer.fit(model)

    trainer = Trainer(tpu_cores=8)
    trainer.fit(model)

6. Can be used as a stand-alone `torch.nn.Module`.

.. code-block:: python

    model = SimCLR()

7. Or use the other parts of the library in your code

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

Or even individual components from models

..code-block:: python

    from pl_bolts.models.autoencoders.basic_ae import AEEncoder
    from pl_bolts.models.autoencoders.basic_vae import Decoder, Encoder
    from pl_bolts.models.self_supervised.cpc import CPCResNet101, CPCTransformsCIFAR10, CPCTransformsImageNet128Patches

-----------------

Community
----------
Bolts is a community driven library! That means all the callbacks, models and weights are contributed
by community members.

To contribute a bolt, just refactor the PyTorch code into Lightning and submit a PR!

--------------------

Modularity
----------
Bolt models and components are built in such a way that each part of the model can be used independently in other
systems. For instance, in the CPC bolt, that system has a special loss function, custom encoders, and even transforms.

If you want to build an extension of that work or use elements from it, just import what you need.

For example, you can just train the full system

.. code-block:: python

    from pl_bolts.models.self_supervised.cpc import CPCV2

    # use as is
    model = CPCV2()

Or use the encoders and transforms from CPC in another system

.. code-block:: python

    from pl_bolts.models.self_supervised.cpc import CPCResNet101, CPCTransformsCIFAR10

--------------

Callbacks
---------
Bolts houses a collection of callbacks that are community contributed and can work in any Lightning Module!

--------------


DataModules
-----------
Bolts also has a collection of datamodules. These allow easy sharing for datasets with
consistent transforms, train, val, tests splits and data preparation steps.

.. code-block:: python

    from pl_bolts.datamodules import MNISTDataLoaders, ImagenetDataModule

    model = LitModel(datamodule=CIFAR10DataLoaders())
    model = LitModel(datamodule=ImagenetDataModule())

We even have prebuilt modules to bridge the gap between Numpy, Sklearn and PyTorch

.. code-block:: python

    from sklearn.datasets import load_boston
    from pl_bolts.datamodules import SklearnDataLoaders

    X, y = load_boston(return_X_y=True)
    datamodule = SklearnDataLoaders(X, y)

    model = LitModel(datamodule)


--------------------

Models
------

Use as a feature extractor
^^^^^^^^^^^^^^^^^^^^^^^^^^
For certain projects that require an architecture you could use this as
a module inside the larger system.

Most models have pretrained weights (usually on Imagenet).

Example::

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (pretrained on Imagenet)
    pretrained_model = VAE(pretrained='imagenet')
    pretrained_model.freeze()

We encourage contributed bolts models to have pretrained weight options as well. For instance, this
resnet18 was trained using self-supervised learning via the CPC approach.

Example::

    from pl_bolts.models.self_supervised import CPCV2

    # feature extractor (pretrained on Imagenet)
    cpc_model = CPCV2(pretrained='resnet18')
    resnet18 = cpc_model.encoder
    resnet18.freeze()

You can also load your own weights after training on your own data.

Example::

    from pl_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    # train
    model = VAE()
    trainer = pl.Trainer()
    trainer.fit(model)

    # feature extractor
    pretrained_model = VAE.load_from_checkpoint(PATH)
    pretrained_model.freeze()

----------------

Use for fine-tuning
^^^^^^^^^^^^^^^^^^^
Can fine-tune on your own data. Either for stand-alone PyTorch

Example::

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (not frozen)
    pretrained_model = VAE.load_from_checkpoint(PATH)

Or in a Lightning Module

Example::

    class YourResearchModel(pl.LightningModule):
        def __init__(self):

            # pretrained VAE
            self.vae = VAE.load_from_checkpoint(PATH)
            self.vae.freeze()

            self.some_other_model = MyModel()

        def forward(self, z):
            # unfreeze at some point
            if self.current_epoch == 10:
                self.vae.unfreeze()

            # generate a sample from z ~ N(0,1)
            x = self.vae(z)

            # do stuff with sample
            x = self.some_other_model(x)
            return x

----------------

Production or for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^
For production or predictions, load weights, freeze the model and use as needed.

Example::

    from pl_bolts.models.autoencoders import VAE

    vae = VAE.load_from_checkpoint(PATH)
    vae.freeze()

    z = ... # z ~ N(0, 1)
    predictions = vae(z)

Train from scratch
^^^^^^^^^^^^^^^^^^
Here's an example on how to train this model from scratch

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE
    import pytorch_lightning as pl

    vae = VAE()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(vae)

----------------

Research
--------
Bolts are designed to be highly configurable and modular.
Here are a few examples showing potential uses in the context of research.

Ex: Changing priors
^^^^^^^^^^^^^^^^^^^
You might be interested in changing the prior of a VAE

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def init_prior(self, z_mu, z_std):
            P = MyPriorDistribution
            # default is standard normal
            # P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
            return P

        def init_posterior(self, z_mu, z_std):
            Q = MyPosteriorDistribution
            # default is normal(z_mu, z_sigma)
            # Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
            return Q

Ex: Changing encoders
^^^^^^^^^^^^^^^^^^^^^
To change parts of the model (for instance, the encoder or decoder) you could do this

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    class MyVAEFlavor(VAE):

        def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
            encoder = MyEncoder(...)
            return encoder

        def init_decoder(self, hidden_dim, latent_dim, input_width, input_height):
            decoder = MyDecoder(...)
            return decoder

Ex: Changing optimizer
^^^^^^^^^^^^^^^^^^^^^^
Every bolt is a Lightning module. This means you can modify anything, even the optimizer used.

Example::

    from pl_bolts.models.autoencoders import VAE

    class MyVAE(VAE):

        def configure_optimizers(self):
            return ANOptimizer(...), OrASecondOne(...)

Ex: Custom backward pass
^^^^^^^^^^^^^^^^^^^^^^^^
Again, just a Lightning Module

Example::

    from pl_bolts.models.self_supervised import CPCV2

    class MyCPC(CPCV2):

        def backward(self):
            # do something weird

Ex: Share components
^^^^^^^^^^^^^^^^^^^^
Bolts are implemented to be modular so parts of these models can be shared across projects

Example::

    from pl_bolts.models.self_supervised.cpc import CPCResNet101, CPCTransformsCIFAR10
    from pl_bolts.models.self_supervised import SimCLR

    class MySimCLR(SimCLR):

        def __init__(self):
            self.encoder = CPCResNet101()

--------------

Production
----------
A major benefit of bolts is that most models have pretrained weights on whatever major datasets
exist for those domains. These weights can be contributed by the community, so the models can be
more domain specific.

.. code-block:: python

    from pl_bolts.models.self_supervised import CPCV2

    # feature extractor (pretrained on Imagenet)
    cpc_model = CPCV2(pretrained='resnet18')
    resnet18 = cpc_model.encoder
    resnet18.freeze()

Even more simple models like VAEs

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (pretrained on Imagenet)
    pretrained_model = VAE(pretrained='imagenet')
    pretrained_model.freeze()

----------------

Regression Heroes
-----------------
In case your job or research doesn't need a "hammer", we offer implementations of Classic ML models
which benefit from lightning's multi-GPU and TPU support. So, now you can run huge workloads
scalably, without needing to do much engineering

Linear Regression
^^^^^^^^^^^^^^^^^
Here's an example for Linear regression

.. code-block:: python

    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataLoaders
    from sklearn.datasets import load_boston

    # link the numpy dataset to PyTorch
    X, y = load_boston(return_X_y=True)
    loaders = SklearnDataLoaders(X, y)

    # training runs training batches while validating against a validation set
    model = LinearRegression()
    trainer = pl.Trainer(num_gpus=8)
    trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())

Once you're done, you can run the test set if needed.

.. code-block:: python

    trainer.test(test_dataloaders=loaders.test_dataloader())

But more importantly, you can scale up to many GPUs, TPUs or even CPUs

.. code-block:: python

    # 8 GPUs
    trainer = pl.Trainer(num_gpus=8)

    # 8 TPUs
    trainer = pl.Trainer(tpu_cores=8)

    # 32 GPUs
    trainer = pl.Trainer(num_gpus=8, num_nodes=4)

    # 128 CPUs
    trainer = pl.Trainer(num_processes=128)

----------------

Regular PyTorch
---------------
Everything in bolts also works with regular PyTorch since they are all just nn.Modules!
However, if you train using Lightning you don't have to deal with engineering code :)

Command line support
--------------------
Any bolt module can also be trained from the command line

.. code-block:: bash

    cd pl_bolts/models/autoencoders/basic_vae
    python basic_vae_pl_module.py

Each script accepts Argparse arguments for both the lightning trainer and the model

.. code-block:: bash

    python basic_vae_pl_module.py -latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12