How to use models
=================
Models are meant to be "bolted" onto your research or production cases.

Bolts are meant to be used in the following ways

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

----------------

Predicting on your data
-----------------------
Most bolts have pretrained weights trained on various datasets or algorithms. This is useful when you
don't have enough data, time or money to do your own training.

For example, you could use a pretrained VAE to generate features for an image dataset.

.. testcode::

    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    encoder = simclr.encoder
    encoder.eval()

.. code-block:: python

    for (x, y) in own_data:
        features = encoder(x)

The advantage of bolts is that each system can be decomposed and used in interesting ways.
For instance, this resnet50 was trained using self-supervised learning (no labels) on Imagenet, and thus
might perform better than the same resnet50 trained with labels

.. testcode::

    # trained without labels
    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    resnet50_unsupervised = simclr.encoder.eval()

    # trained with labels
    from torchvision.models import resnet50
    resnet50_supervised = resnet50(pretrained=True)

.. code-block:: python

    # perhaps the features when trained without labels are much better for classification or other tasks
    x = image_sample()
    unsup_feats = resnet50_unsupervised(x)
    sup_feats = resnet50_supervised(x)

    # which one will be better?

Bolts are often trained on more than just one dataset.

.. testcode::

    from pl_bolts.models.self_supervised import SimCLR

    # imagenet weights
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    simclr.freeze()

---------------

Finetuning on your data
-----------------------
If you have a little bit of data and can pay for a bit of training, it's often better to finetune on your own data.

To finetune you have two options unfrozen finetuning or unfrozen later.

Unfrozen Finetuning
^^^^^^^^^^^^^^^^^^^
In this approach, we load the pretrained model and unfreeze from the beginning

.. testcode::

    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    resnet50 = simclr.encoder
    # don't call .freeze()

.. code-block:: python

    classifier = LogisticRegression(...)

    for (x, y) in own_data:
        feats = resnet50(x)
        y_hat = classifier(feats)
        ...

Or as a LightningModule

.. code-block:: python

    class FineTuner(pl.LightningModule):

        def __init__(self, encoder):
            self.encoder = encoder
            self.classifier = LogisticRegression(...)

        def training_step(self, batch, batch_idx):
            (x, y) = batch
            feats = self.encoder(x)
            y_hat = self.classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)
            return loss

    trainer = Trainer(gpus=2)
    model = FineTuner(resnet50)
    trainer.fit(model)

Sometimes this works well, but more often it's better to keep the encoder frozen for a while

Freeze then unfreeze
^^^^^^^^^^^^^^^^^^^^
The approach that works best most often is to freeze first then unfreeze later

.. testcode::

    # freeze!
    from pl_bolts.models.self_supervised import SimCLR

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
    resnet50 = simclr.encoder
    resnet50.eval()

.. code-block:: python

    classifier = LogisticRegression(...)

    for epoch in epochs:
        for (x, y) in own_data:
            feats = resnet50(x)
            y_hat = classifier(feats)
            loss = cross_entropy_with_logits(y_hat, y)

        # unfreeze after 10 epochs
        if epoch == 10:
            resnet50.unfreeze()

.. note:: In practice, unfreezing later works MUCH better.

Or in Lightning as a Callback so you don't pollute your research code.

.. code-block:: python

    class UnFreezeCallback(Callback):

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == 10.
                encoder.unfreeze()

    trainer = Trainer(gpus=2, callbacks=[UnFreezeCallback()])
    model = FineTuner(resnet50)
    trainer.fit(model)

Unless you still need to mix it into your research code.

.. code-block:: python

    class FineTuner(pl.LightningModule):

        def __init__(self, encoder):
            self.encoder = encoder
            self.classifier = LogisticRegression(...)

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

    from pl_bolts.models.autoencoders import VAE

    learning_rates = [0.01, 0.001, 0.0001]
    hidden_dim = [128, 256, 512]

    for lr in learning_rates:
        for hd in hidden_dim:
            vae = VAE(input_height=32, hidden_dim=hd, learning_rate=lr)
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
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

.. note:: For this to work well, make sure you have enough data and time to train these models!

-------------

For research
------------
What separates bolts from all the other libraries out there is that bolts is built by and used by AI researchers.
This means every single bolt is modularized so that it can be easily extended or mixed with arbitrary parts of
the rest of the code-base.

Extending work
^^^^^^^^^^^^^^
Perhaps a research project requires modifying a part of a know approach. In this case, you're better off only
changing that part of a system that is already know to perform well. Otherwise, you risk not implementing the work
correctly.

**Example 1: Changing the prior or approx posterior of a VAE**

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

And of course train it with lightning.

.. code-block:: python

    model = MyVAEFlavor()
    trainer = Trainer()
    trainer.fit(model)

In just a few lines of code you changed something fundamental about a VAE... This
means you can iterate through ideas much faster knowing that the bolt implementation and the training loop are CORRECT
and TESTED.

If your model doesn't work with the new P, Q, then you can discard that research idea much faster than trying to
figure out if your VAE implementation was correct, or if your training loop was correct.

**Example 2: Changing the generator step of a GAN**

.. testcode::

    from pl_bolts.models.gans import GAN

    class FancyGAN(GAN):

        def generator_step(self, x):
            # sample noise
            z = torch.randn(x.shape[0], self.hparams.latent_dim)
            z = z.type_as(x)

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all real)
            real = torch.ones(x.size(0), 1)
            real = real.type_as(x)
            g_loss = self.generator_loss(real)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

**Example 3: Changing the way the loss is calculated in a contrastive self-supervised learning approach**

.. testcode::

    from pl_bolts.models.self_supervised import AMDIM

    class MyDIM(AMDIM):

        def validation_step(self, batch, batch_nb):
            [img_1, img_2], labels = batch

            # generate features
            r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2 = self.forward(img_1, img_2)

            # Contrastive task
            loss, lgt_reg = self.contrastive_task((r1_x1, r5_x1, r7_x1), (r1_x2, r5_x2, r7_x2))
            unsupervised_loss = loss.sum() + lgt_reg

            result = {
                'val_nce': unsupervised_loss
            }
            return result

---------------

Importing parts
^^^^^^^^^^^^^^^
All the bolts are modular. This means you can also arbitrarily mix and match fundamental blocks from across
approaches.

**Example 1: Use the VAE encoder for a GAN as a generator**

.. code-block:: python

    from pl_bolts.models.gans import GAN
    from pl_bolts.models.autoencoders.basic_vae import Encoder

    class FancyGAN(GAN):

        def init_generator(self, img_dim):
            generator = Encoder(...)
            return generator

    trainer = Trainer(...)
    trainer.fit(FancyGAN())

**Example 2: Use the contrastive task of AMDIM in CPC**

.. testcode::

    from pl_bolts.models.self_supervised import AMDIM, CPC_v2

    default_amdim_task = AMDIM().contrastive_task
    model = CPC_v2(contrastive_task=default_amdim_task, encoder='cpc_default')
    # you might need to modify the cpc encoder depending on what you use

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   ...

---------------

Compose new ideas
^^^^^^^^^^^^^^^^^
You may also be interested in creating completely new approaches that mix and match all sorts of different
pieces together

.. testcode::

    # this model is for illustration purposes, it makes no research sense but it's intended to show
    # that you can be as creative and expressive as you want.
    class MyNewContrastiveApproach(pl.LightningModule):

        def __init__(self):
            suoer().__init_()

            self.gan = GAN()
            self.vae = VAE()
            self.amdim = AMDIM()
            self.cpc = CPC_v2

        def training_step(self, batch, batch_idx):
            (x, y) = batch

            feat_a = self.gan.generator(x)
            feat_b = self.vae.encoder(x)

            unsup_loss = self.amdim(feat_a) + self.cpc(feat_b)

            vae_loss = self.vae._step(batch)
            gan_loss = self.gan.generator_loss(x)

            return unsup_loss + vae_loss + gan_loss
