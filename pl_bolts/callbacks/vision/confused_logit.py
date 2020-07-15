import torch
from torch import nn
from matplotlib import pyplot as plt
from pytorch_lightning import Callback


class ConfusedLogitCallback(Callback):

    def __init__(self, top_k, projection_factor=3):
        """
        Takes the logit predictions of a model and when the probabilities of two classes are very close, the model
        doesn't have high certainty that it should pick one vs the other class.

        This callback shows how the input would have to change to swing the model from one label prediction
        to the other.

        .. note:: whenever called, this model will look for self.last_batch and self.last_logits in the LightningModule

        .. note:: this callback supports tensorboard only right now

        Args:
            top_k: How many "offending" images we should plot
            projection_factor: How much to multiply the input image to make it look more like this logit label
        """
        super().__init__()
        self.top_k = top_k
        self.projection_factor = projection_factor

    def on_batch_end(self, trainer, pl_module):

        # show images only every 20 batches
        if (trainer.batch_idx + 1) % 20 != 0:
            return

        # pick the last batch and logits
        # TODO: use context instead
        x, y = pl_module.last_batch
        l = pl_module.last_logits

        # only check when it has opinions (ie: the logit > 5)
        if l.max() > 5.0:
            # pick the top two confused probs
            (values, idxs) = torch.topk(l, k=2, dim=1)

            # care about only the ones that are at most eps close to each other
            eps = 0.1
            mask = (values[:, 0] - values[:, 1]).abs() < eps

            if mask.sum() > 0:
                # pull out the ones we care about
                confusing_x = x[mask, ...]
                confusing_y = y[mask]
                confusing_l = l[mask]

                mask_idxs = idxs[mask]

                self._plot(confusing_x, confusing_y, trainer, pl_module, mask_idxs)

    def _plot(self, confusing_x, confusing_y, trainer, model, mask_idxs):
        batch_size = confusing_x.size(0)

        confusing_x = confusing_x[:self.top_k]
        confusing_y = confusing_y[:self.top_k]

        model.eval()
        x_param_a = nn.Parameter(confusing_x)
        x_param_b = nn.Parameter(confusing_x)

        for logit_i, x_param in enumerate((x_param_a, x_param_b)):
            l = model(x_param.view(batch_size, -1))
            l[:, mask_idxs[:, logit_i]].sum().backward()

        # reshape grads
        grad_a = x_param_a.grad.view(batch_size, 28, 28)
        grad_b = x_param_b.grad.view(batch_size, 28, 28)

        for img_i in range(len(confusing_x)):
            x = confusing_x[img_i].squeeze(0)
            y = confusing_y[img_i]
            ga = grad_a[img_i]
            gb = grad_b[img_i]

            mask_idx = mask_idxs[img_i]

            fig = plt.figure(figsize=(15, 10))
            plt.subplot(231)
            plt.imshow(x)
            plt.colorbar()
            plt.title(f'True: {y}', fontsize=20)

            plt.subplot(232)
            plt.imshow(ga)
            plt.colorbar()
            plt.title(f'd{mask_idx[0]}-logit/dx', fontsize=20)

            plt.subplot(233)
            plt.imshow(gb)
            plt.colorbar()
            plt.title(f'd{mask_idx[1]}-logit/dx', fontsize=20)

            plt.subplot(235)
            plt.imshow(ga * 2 + x)
            plt.colorbar()
            plt.title(f'd{mask_idx[0]}-logit/dx', fontsize=20)

            plt.subplot(236)
            plt.imshow(gb * 2 + x)
            plt.colorbar()
            plt.title(f'd{mask_idx[1]}-logit/dx', fontsize=20)

            trainer.logger.experiment.add_figure('confusing_imgs', fig, global_step=trainer.global_step)
