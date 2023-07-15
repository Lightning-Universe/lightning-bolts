Introduction Guide
==================
Welcome to PyTorch Lightning Bolts!

Bolts is a Deep learning research and production toolbox of:

- SOTA pretrained models.
- Model components.
- Callbacks.
- Losses.
- Datasets.

**The Main goal of Bolts is to enable trying new ideas as fast as possible!**

.. note:: Currently, Bolts is going through a major revision. For more information about it, see these GitHub issues (`#819 <https://github.com/Lightning-AI/lightning-bolts/issues/819>`_ and `#839 <https://github.com/Lightning-AI/lightning-bolts/issues/839>`_) and `stability section <https://lightning-bolts.readthedocs.io/en/latest/stability.html>`_

All models are tested (daily), benchmarked, documented and work on CPUs, TPUs, GPUs and 16-bit precision.

**some examples!**

.. code-block:: python

    from pl_bolts.models import VAE
    from pl_bolts.models.vision import GPT2, ImageGPT, PixelCNN
    from pl_bolts.models.self_supervised import AMDIM, CPC_v2, SimCLR, MoCo
    from pl_bolts.models import LinearRegression, LogisticRegression
    from pl_bolts.models.gans import GAN
    from pl_bolts.callbacks import PrintTableMetricsCallback
    from pl_bolts.datamodules import FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule

**Bolts are built for rapid idea iteration - subclass, override and train!**

.. code-block:: python

    from pl_bolts.models.vision import ImageGPT
    from pl_bolts.models.self_supervised import SimCLR

    class VideoGPT(ImageGPT):

        def training_step(self, batch, batch_idx):
            x, y = batch
            x = _shape_input(x)

            logits = self.gpt(x)
            simclr_features = self.simclr(x)

            # -----------------
            # do something new with GPT logits + simclr_features
            # -----------------

            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

            logs = {"loss": loss}
            return {"loss": loss, "log": logs}

**Mix and match data, modules and components as you please!**

.. code-block:: python

    model = GAN(datamodule=ImagenetDataModule(PATH))
    model = GAN(datamodule=FashionMNISTDataModule(PATH))
    model = ImageGPT(datamodule=FashionMNISTDataModule(PATH))

**And train on any hardware accelerator**

.. code-block:: python

    import pytorch_lightning as pl

    model = ImageGPT(datamodule=FashionMNISTDataModule(PATH))

    # cpus
    pl.Trainer.fit(model)

    # gpus
    pl.Trainer(gpus=8).fit(model)

    # tpus
    pl.Trainer(tpu_cores=8).fit(model)

**Or pass in any dataset of your choice**

.. code-block:: python

    model = ImageGPT()
    Trainer().fit(
        model,
        train_dataloader=DataLoader(...),
        val_dataloader=DataLoader(...)
    )

-------------

Community Built
---------------
Then lightning community builds bolts and contributes them to Bolts.
The lightning team guarantees that contributions are:

1. Rigorously tested (CPUs, GPUs, TPUs).
2. Rigorously documented.
3. Standardized via PyTorch Lightning.
4. Optimized for speed.
5. Checked for correctness.

-------------

How to contribute
^^^^^^^^^^^^^^^^^
We accept contributions directly to Bolts or via your own repository.

.. note:: We encourage you to have your own repository so we can link to it via our docs!

To contribute:

1. Submit a pull request to Bolts (we will help you finish it!).
2. We'll help you add `tests <https://github.com/PyTorchLightning/lightning-bolts/tree/master/tests>`_.
3. We'll help you refactor models to work on `(GPU, TPU, CPU). <https://www.youtube.com/watch?v=neuNEcN9FK4>`_.
4. We'll help you remove bottlenecks in your model.
5. We'll help you write up `documentation <https://lightning-bolts.readthedocs.io/en/latest/convolutional.html#image-gpt>`_.
6. We'll help you pretrain expensive models and host weights for you.
7. We'll create proper attribution for you and link to your repo.
8. Once all of this is ready, we will merge into bolts.


After your model or other contribution is in bolts, our team will make sure it maintains compatibility
with the other components of the library!

---------------

Contribution ideas
^^^^^^^^^^^^^^^^^^
Don't have something to contribute? Ping us on
`Slack <https://www.pytorchlightning.ai/community>`_
or look at our `Github issues <https://github.com/PyTorchLightning/lightning-bolts/
issues?q=is%3Aissue+is%3Aopen+label%3A%22Model+to+implement%22>`_!

**We'll help and guide you through the implementation / conversion**

---------------

When to use Bolts
-----------------

For pretrained models
^^^^^^^^^^^^^^^^^^^^^
Most bolts have pretrained weights trained on various datasets or algorithms. This is useful when you
don't have enough data, time or money to do your own training.

For example, you could use a pretrained VAE to generate features for an image dataset.

.. testcode::

    from pl_bolts.models.autoencoders import VAE
    from pl_bolts.models.self_supervised import CPC_v2

    model1 = VAE(input_height=32, pretrained='imagenet2012')
    encoder = model1.encoder
    encoder.eval()

    # bolts are pretrained on different datasets
    model2 = CPC_v2(encoder='resnet18', pretrained='imagenet128').freeze()
    model3 = CPC_v2(encoder='resnet18', pretrained='stl10').freeze()

.. code-block:: python

    for (x, y) in own_data:
        features = encoder(x)
        feat2 = model2(x)
        feat3 = model3(x)

    # which is better?

To finetune on your data
^^^^^^^^^^^^^^^^^^^^^^^^
If you have your own data, finetuning can often increase the performance. Since this is pure PyTorch
you can use any finetuning protocol you prefer.

**Example 1: Unfrozen finetune**

.. code-block:: python

    # unfrozen finetune
    model = CPC_v2(encoder='resnet18', pretrained='imagenet128')
    resnet18 = model.encoder
    # don't call .freeze()

    classifier = LogisticRegression(...)

    for (x, y) in own_data:
        feats = resnet18(x)
        y_hat = classifier(feats)

**Example 2: Freeze then unfreeze**

.. code-block:: python

    # FREEZE!
    model = CPC_v2(encoder='resnet18', pretrained='imagenet128')
    resnet18 = model.encoder
    resnet18.eval()

    classifier = LogisticRegression(...)

    for epoch in epochs:
        for (x, y) in own_data:
            feats = resnet18(x)
            y_hat = classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)

        # UNFREEZE after 10 epochs
        if epoch == 10:
            resnet18.unfreeze()

For research
^^^^^^^^^^^^
Here is where bolts is very different than other libraries with models. It's not just designed
for production, but each module is written to be easily extended for research.

.. code-block:: python

    from pl_bolts.models.vision import ImageGPT
    from pl_bolts.models.self_supervised import SimCLR

    class VideoGPT(ImageGPT):

        def training_step(self, batch, batch_idx):
            x, y = batch
            x = _shape_input(x)

            logits = self.gpt(x)
            simclr_features = self.simclr(x)

            # -----------------
            # do something new with GPT logits + simclr_features
            # -----------------

            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

            logs = {"loss": loss}
            return {"loss": loss, "log": logs}

Or perhaps your research is in self_supervised_learning and you want to do a new SimCLR. In this case, the only
thing you want to change is the loss.

By subclassing you can focus on changing a single piece of a system without worrying that the other parts work
(because if they are in Bolts, then they do and we've tested it).

.. code-block:: python

    # subclass SimCLR and change ONLY what you want to try
    class ComplexCLR(SimCLR):

        def init_loss(self):
            return self.new_xent_loss

        def new_xent_loss(self):
            out = torch.cat([out_1, out_2], dim=0) n_samples = len(out)

            # Full similarity matrix
            cov = torch.mm(out, out.t().contiguous())
            sim = torch.exp(cov / temperature)

            # Negative similarity
            mask = ~torch.eye(n_samples, device=sim.device).bool()
            neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

            # ------------------
            # some new thing we want to do
            # ------------------

            # Positive similarity :
            pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos = torch.cat([pos, pos], dim=0)
            loss = -torch.log(pos / neg).mean()

            return loss

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
In PyTorch, working with data has these major elements.

    1. Downloading, saving and preparing the dataset.
    2. Splitting into train, val and test.
    3. For each split, applying different transforms

A DataModule groups together those actions into a single reproducible `DataModule` that can be shared
around to guarantee:

    1. Consistent data preprocessing (download, splits, etc...)
    2. The same exact splits
    3. The same exact transforms

.. code-block:: python

    from pl_bolts.datamodules import ImagenetDataModule

    dm = ImagenetDataModule(data_dir=PATH)

    # standard PyTorch!
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    Trainer().fit(
        model,
        train_loader,
        val_loader
    )


But when paired with PyTorch LightningModules (all bolts models), you can plug and play
full dataset definitions with the same splits, transforms, etc...


.. code-block:: python

    imagenet = ImagenetDataModule(PATH)
    model = VAE(datamodule=imagenet)
    model = ImageGPT(datamodule=imagenet)
    model = GAN(datamodule=imagenet)



We even have prebuilt modules to bridge the gap between Numpy, Sklearn and PyTorch

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from pl_bolts.datamodules import SklearnDataModule

    X, y = load_diabetes(return_X_y=True)
    datamodule = SklearnDataModule(X, y)

    model = LitModel(datamodule)

---------------

Regression Heroes
-----------------
In case your job or research doesn't need a "hammer", we offer implementations of Classic ML models
which benefit from lightning's multi-GPU and TPU support.

So, now you can run huge workloads scalably, without needing to do any engineering.
For instance, here we can run logistic Regression on Imagenet (each epoch takes about 3 minutes)!

.. code-block:: python

    from pl_bolts.models.regression import LogisticRegression

    imagenet = ImagenetDataModule(PATH)

    # 224 x 224 x 3
    pixels_per_image = 150528
    model = LogisticRegression(input_dim=pixels_per_image, num_classes=1000)
    model.prepare_data = imagenet.prepare_data

    trainer = Trainer(gpus=2)
    trainer.fit(
        model,
        imagenet.train_dataloader(batch_size=256),
        imagenet.val_dataloader(batch_size=256)
    )

Linear Regression
^^^^^^^^^^^^^^^^^
Here's an example for Linear regression

.. code-block:: python

    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataModule
    from sklearn.datasets import load_diabetes

    # link the numpy dataset to PyTorch
    X, y = load_diabetes(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    # training runs training batches while validating against a validation set
    model = LinearRegression()
    trainer = pl.Trainer(num_gpus=8)
    trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())

Once you're done, you can run the test set if needed.

.. code-block:: python

    trainer.test(test_dataloaders=loaders.test_dataloader())

But more importantly, you can scale up to many GPUs, TPUs or even CPUs

.. code-block:: python

    # 8 GPUs
    trainer = pl.Trainer(num_gpus=8)

    # 8 TPU cores
    trainer = pl.Trainer(tpu_cores=8)

    # 32 GPUs
    trainer = pl.Trainer(num_gpus=8, num_nodes=4)

    # 128 CPUs
    trainer = pl.Trainer(num_processes=128)

Logistic Regression
^^^^^^^^^^^^^^^^^^^
Here's an example for logistic regression

.. code-block:: python

    from sklearn.datasets import load_iris
    from pl_bolts.models.regression import LogisticRegression
    from pl_bolts.datamodules import SklearnDataModule
    import pytorch_lightning as pl

    # use any numpy or sklearn dataset
    X, y = load_iris(return_X_y=True)
    dm = SklearnDataModule(X, y, batch_size=12)

    # build model
    model = LogisticRegression(input_dim=4, num_classes=3)

    # fit
    trainer = pl.Trainer(tpu_cores=8, precision=16)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    trainer.test(test_dataloaders=dm.test_dataloader())

Any input will be flattened across all dimensions except the first one (batch).
This means images, sound, etc... work out of the box.

.. code-block:: python

    # create dataset
    dm = MNISTDataModule(num_workers=0, data_dir=tmpdir)

    model = LogisticRegression(input_dim=28 * 28, num_classes=10, learning_rate=0.001)
    model.prepare_data = dm.prepare_data
    model.train_dataloader = dm.train_dataloader
    model.val_dataloader = dm.val_dataloader
    model.test_dataloader = dm.test_dataloader

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model)
    trainer.test(model)
    # {test_acc: 0.92}

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

    python basic_vae_pl_module.py --latent_dim 32 --batch_size 32 --gpus 4 --max_epochs 12
