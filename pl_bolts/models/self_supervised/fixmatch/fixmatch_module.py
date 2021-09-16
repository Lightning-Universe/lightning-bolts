import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

from .lr_scheduler import WarmupCosineLrScheduler
from .networks import WideResnet, ema_model_update, get_ema_model


class FixMatch(LightningModule):
    def __init__(
            self,
            ema_eval: bool = True,
            batch_size: int = 16,
            mu: int = 7,
            wresnet_k: int = 8,
            wresnet_n: int = 28,
            ema_decay: float = 0.999,
            softmax_temperature: float = 1.0,
            distribution_alignment: bool = True,
            coefficient_unsupervised: float = 1.0,
            pseudo_thr: float = 0.95,
            lr: float = 0.03,
            weight_decay: float = 1e-3,
            momentum: float = 0.9,
            gpus: int = 1,
            max_epochs: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ema_eval = ema_eval
        self.mu = mu
        self.batch_size = batch_size
        self.wresnet_k = wresnet_k
        self.wresnet_n = wresnet_n
        self.ema_decay = ema_decay
        self.softmax_temperature = softmax_temperature
        self.distribution_alignment = distribution_alignment
        self.coefficient_unsupervised = coefficient_unsupervised
        self.pseudo_thr = pseudo_thr
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gpus = gpus
        self.max_epochs = max_epochs
        self.criteria_x = nn.CrossEntropyLoss()
        self.criteria_u = nn.CrossEntropyLoss(reduction="none")
        self.prob_list = []

    @staticmethod
    def interleave(x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def de_interleave(x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        if stage == "fit":
            train_loader = self.train_dataloader()
            self.n_classes = len(train_loader["labeled"].dataset.classes)
            self.model = WideResnet(n_classes=self.n_classes, k=self.wresnet_k, n=self.wresnet_n)
            if self.ema_eval:
                self.ema_model = get_ema_model(self.model)
            self.total_steps = (len(train_loader["labeled"].dataset) // (self.batch_size * max(1, self.gpus))) * float(
                self.max_epochs
            )

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]  # X
        unlabeled_batch = batch["unlabeled"]  # U

        img_x_weak, label_x = labeled_batch
        (img_u_weak, img_u_strong), label_u = unlabeled_batch

        batch_size = img_x_weak.size(0)
        imgs = self.interleave(torch.cat([img_x_weak, img_u_weak, img_u_strong]), 2 * self.mu + 1)
        logits = self.model(imgs)
        logits = self.de_interleave(logits, 2 * self.mu + 1)
        logits_x = logits[:batch_size]
        logits_u_weak, logits_u_strong = logits[batch_size:].chunk(2)
        del logits

        supervised_loss = self.criteria_x(logits_x, label_x)
        with torch.no_grad():
            probs = self.get_unlabled_logits_weak_probs(logits_u_weak)
            mask, label_u_guess = self.get_pesudo_mask_and_infer_u_label(probs)

        unsupervised_loss = (self.criteria_u(logits_u_strong, label_u_guess) * mask).mean()

        loss = supervised_loss + self.coefficient_unsupervised * unsupervised_loss
        # ema eval
        if self.ema_eval:
            with torch.no_grad():
                ema_model_update(self.model, self.ema_model, self.ema_decay)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/supervised_loss", supervised_loss, on_step=True, on_epoch=True)
        self.log("train/unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=True)
        corr_u_label = (label_u_guess == label_u).float() * mask
        self.log("train/acc@unlabeled", corr_u_label.sum().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc@strong", mask.sum().item(), on_step=True, on_epoch=True)
        self.log("train/mask", mask.mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def eval_forward(self, images):
        return self.model(images)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.eval_forward(images)
        loss = self.criteria_x(logits, labels)
        acc1, acc5 = self.__accuracy(logits, labels, topk=(1, 5))
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/acc@1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val/acc@5", acc5, on_step=True, on_epoch=True)
        if self.ema_eval:
            ema_logits = self.ema_model(images)
            ema_loss = self.criteria_x(ema_logits, labels)
            ema_acc1, ema_acc5 = self.__accuracy(ema_logits, labels, topk=(1, 5))
            self.log("val/ema_loss", ema_loss, on_step=True, on_epoch=True)
            self.log("val/ema_acc1", ema_acc1, on_step=True, on_epoch=True)
            self.log("val/ema_acc5", ema_acc5, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "bn"]
        grouped_params = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0},
        ]
        optimizer = optim.SGD(
            grouped_params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        scheduler = WarmupCosineLrScheduler(optimizer, max_iter=self.total_steps, warmup_iter=0)
        return [optimizer], [scheduler]

    def get_unlabled_logits_weak_probs(self, logits_u_weak):
        probs = torch.softmax(logits_u_weak / self.softmax_temperature, dim=1)
        if self.distribution_alignment:
            self.prob_list.append(probs.mean(0))
            if len(self.prob_list) > 32:
                self.prob_list.pop(0)
            prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def get_pesudo_mask_and_infer_u_label(self, probs):
        scores, label_u_guess = torch.max(probs, dim=1)
        mask = scores.ge(self.pseudo_thr).float()
        return mask, label_u_guess

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


class FixMatchCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Link args.
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("trainer.gpus", "model.gpus")
        parser.link_arguments("data.mu", "model.mu")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")


if __name__ == "__main__":
    from pl_bolts.models.self_supervised.fixmatch.datasets import SSLDataModule

    cli = FixMatchCLI(FixMatch, SSLDataModule)
