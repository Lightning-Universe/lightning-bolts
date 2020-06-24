How to use models
=================
Models are meant to be "bolted" onto your research or production cases.

Bolts are meant to be used in the following ways

----------------

Predicting on your data
-----------------------
Most bolts have pretrained weights trained on various datasets or algorithms. This is useful when you
don't have enough data, time or money to do your own training.

For example, you could use a pretrained VAE to generate features for an image dataset.

.. code-block:: python

    from pl_bolts.models.autoencoders import VAE

    model = VAE(pretrained='imagenet2012')
    encoder = model.encoder
    encoder.freeze()

    for (x, y) in own_data
        features = encoder(x)

The advantage of bolts is that each system can be decomposed and used in interesting ways.
For instance, this resnet18 was trained using self-supervised learning (no labels) on Imagenet, and thus
might perform better than the same resnet18 trained with labels

.. code-block:: python

    # trained without labels
    from pl_bolts.models.self_supervised import CPCV2

    model = CPCV2(encoder='resnet18', pretrained='imagenet128')
    resnet18_unsupervised = model.encoder.freeze()

    # trained with labels
    from torchvision.models import resnet18
    resnet18_supervised = resnet18(pretrained=True)

    # perhaps the features when trained without labels are much better for classification or other tasks
    x = image_sample()
    unsup_feats = resnet18_unsupervised(x)
    sup_feats = resnet18_supervised(x)

    # which one will be better?

Bolts are often trained on more than just one dataset.

.. code-block:: python

    model = CPCV2(encoder='resnet18', pretrained='stl10')


---------------

Finetuning on your data
-----------------------
If you have a little bit of data and can pay for a bit of training, it's often better to finetune on your own data.

To finetune you have two options unfrozen finetuning or unfrozen later.

Unfrozen Finetuning
^^^^^^^^^^^^^^^^^^^
In this approach, we load the pretrained model and unfreeze from the beginning

.. code-block:: python

    model = CPCV2(encoder='resnet18', pretrained='imagenet128')
    resnet18 = model.encoder
    # don't call .freeze()

    classifier = LogisticRegression()

    for (x, y) in own_data:
        feats = resnet18(x)
        y_hat = classifier(feats)
        ...

Or as a LightningModule

.. code-block:: python

    class FineTuner(pl.LightningModule):

        def __init__(self, encoder):
            self.encoder = encoder
            self.classifier = LogisticRegression()

        def training_step(self, batch, batch_idx):
            (x, y) = batch
            feats = self.encoder(x)
            y_hat = self.classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)
            return loss

    trainer = Trainer(gpus=2)
    model = FineTuner(resnet18)
    trainer.fit(model)

Sometimes this works well, but more often it's better to keep the encoder frozen for a while

Freeze then unfreeze
^^^^^^^^^^^^^^^^^^^^
The approach that works best most often is to freeze first then unfreeze later

.. code-block:: python

    # freeze!
    model = CPCV2(encoder='resnet18', pretrained='imagenet128')
    resnet18 = model.encoder
    resnet18.freeze()

    classifier = LogisticRegression()

    for epoch in epochs:
        for (x, y) in own_data:
            feats = resnet18(x)
            y_hat = classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)

        # unfreeze after 10 epochs
        if epoch == 10:
            resnet18.unfreeze()

.. note:: In practice, unfreezing later works MUCH better.

Or in Lightning as a Callback so you don't pollute your research code.

.. code-block:: python

    class UnFreezeCallback(Callback):

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == 10.
                encoder.unfreeze()

    trainer = Trainer(gpus=2, callbacks=[UnFreezeCallback()])
    model = FineTuner(resnet18)
    trainer.fit(model)

Unless you still need to mix it into your research code.

.. code-block:: python

    class FineTuner(pl.LightningModule):

        def __init__(self, encoder):
            self.encoder = encoder
            self.classifier = LogisticRegression()

        def training_step(self, batch, batch_idx):

            # option 1 - (not recommended because it's messy)
            if self.trainer.current_epoch == 10:
                self.encoder.unfreeze()

            (x, y) = batch
            feats = self.encoder(x)
            y_hat = self.classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)
            return loss

        def on_epoch_end(self, trainer, pl_module):
            # a hook is cleaner (but a callback is much better)
            if self.trainer.current_epoch == 10:
                self.encoder.unfreeze()


Hyperparameter search
^^^^^^^^^^^^^^^^^^^^^
For finetuning to work well, you should try many versions of the model hyperparameters. Otherwise you're unlikely
to get the most value out of your data.

.. code-block:: python

    learning_rates = [0.01, 0.001, 0.0001]
    hidden_dim = [128, 256, 512]

    for lr in learning_rates:
        for hd in hidden_dim:
            vae = VAE(hidden_dim=hd, learning_rate=lr)
            trainer = Trainer()
            trainer.fit(vae)

--------------

Train from scratch
------------------
If you do have enough data and compute resources, then you could try training from scratch.

.. code-block:: python

    # get data
    train_data = DataLoader(YourDataset)
    val_data = DataLoader(YourDataset)

    # use any bolts model without pretraining
    model = VAE()

    # fit!
    trainer = Trainer(gpus=2)
    trainer.fit(model, train_data, val_data)

.. note:: For this to work well, make sure you have enough data and time to train these models!
