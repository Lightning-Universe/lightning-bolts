Introduction Guide
==================
Welcome to bolts - The community collection of Lightning models, callbacks and modules which can be "bolted" onto
PyTorch projects.

Bolts are built by, and added by the community. But unlike repos with one-off models,
the lightning team guarantees that bolts are:

1. Tested
2. Standardized
3. Documented
4. Curated
5. Well maintained
6. Checked for correctness

-------------

What is it?
-----------
Lightning Bolts is a collection of Models, Callbacks and other goodies implemented in PyTorch Lightning.

Bolts are designed to bootstrap research or to be used in production. Here are ways in which bolts can be used

**As a feature extractor for production and research systems.**

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    # feature extractor (pretrained on Imagenet)
    pretrained_model = VAE(pretrained='imagenet')
    pretrained_model.freeze()

**Subclass and override to try new research ideas.**

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


**Fully pre-built models that can be FINE-TUNED on your data**

.. code-block:: python

    from pl_bolts.models.self_supervised import CPCV2

    # feature extractor (pretrained on Imagenet)
    cpc_model = CPCV2(encoder='resnet18', pretrained='imagenet')
    resnet18 = cpc_model.encoder
    resnet18.freeze()

**Fully contained algorithms that can be TRAINED on your data from scratch**

.. code-block:: python

    from pl_bolts.models.self_supervised import SimCLR

    model = SimCLR()
    train_data = DataLoader(yourData())
    val_data = DataLoader(yourData())

    trainer = Trainer()
    trainer.fit(model, train_data, val_data)

**Fully compatible with GPUs, TPUs, 16-bit precision, etc because of PyTorch Lightning**

.. code-block:: python

    model = SimCLR()

    trainer = Trainer(num_nodes=8, gpus=8)
    trainer.fit(model)

    trainer = Trainer(tpu_cores=8)
    trainer.fit(model)

**Can be used as a stand-alone `torch.nn.Module`**

.. code-block:: python

    model = SimCLR()

**Or use the other parts of the library in your code**

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

**Or even individual components from models**

.. code-block:: python

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
Callbacks are arbitrary programs which can run at any points in time within a training loop in Lightning.

Bolts houses a collection of callbacks that are community contributed and can work in any Lightning Module!

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback
    import pytorch_lightning as pl

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

--------------

DataModules
-----------
A DataModule abstracts away all the details of train, test, val splits, downloading data
and processing the data. DataModules can be shared and dropped into LightningModules.

Bolt has a collection of these modules built by the community.

.. code-block:: python

    from pl_bolts.datamodules import MNISTDataModule, ImagenetDataModule

    model = LitModel(datamodule=CIFAR10DataModule())
    model = LitModel(datamodule=ImagenetDataModule())


DataModules are just collections of train, val and test DataLoaders. This means
you can also use them without lightning

.. code-block:: python

    imagenet = ImagenetDataModule()

    for batch in imagenet.train_dataloader():
        ...
        for val_batch in imagenet.val_dataloader()
        ...

    for test_batch in imagenet.test_dataloader():
        ...


We even have prebuilt modules to bridge the gap between Numpy, Sklearn and PyTorch

.. code-block:: python

    from sklearn.datasets import load_boston
    from pl_bolts.datamodules import SklearnDataModule

    X, y = load_boston(return_X_y=True)
    datamodule = SklearnDataModule(X, y)

    model = LitModel(datamodule)


--------------------

.. include:: models.rst

---------------

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
    from pl_bolts.datamodules import SklearnDataModule
    from sklearn.datasets import load_boston

    # link the numpy dataset to PyTorch
    X, y = load_boston(return_X_y=True)
    loaders = SklearnDataModule(X, y)

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

----------------

Command line support
--------------------
Any bolt module can also be trained from the command line

.. code-block:: bash

    cd pl_bolts/models/autoencoders/basic_vae
    python basic_vae_pl_module.py

Each script accepts Argparse arguments for both the lightning trainer and the model

.. code-block:: bash

    python basic_vae_pl_module.py -latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12