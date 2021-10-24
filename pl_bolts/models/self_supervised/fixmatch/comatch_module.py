import torch
from pytorch_lightning.utilities.cli import LightningCLI

from pl_bolts.models.self_supervised.fixmatch.fixmatch_module import FixMatch, WideResnet


class Queue:
    def __init__(self, value, size):
        self.value = value
        self.size = size
        self.ptr = 0

    def __call__(self):
        return self.value

    def update(self, new_value, total_batch_size):
        if new_value.device != self.value.device:
            # Move to the updated device.
            self.value = self.value.to(new_value.device)
        self.value[self.ptr : self.ptr + total_batch_size, :] = new_value
        self.ptr = (self.ptr + total_batch_size) % self.size


class CoMatch(FixMatch):
    """PyTorch Lightning implementation of CoMatch: Semi-supervised Learning with Contrastive Graph Regularization

        Paper authors: Junnan Li, Caiming Xiong, Steven Hoi

        Model implemented by: `Zehua Cheng <https://github.com/limberc>`_

        This code is adapted to Lightning using the original author repo
        (`the original repo <https://github.com/salesforce/CoMatch/>`_).
        Original work is: Copyright (c) 2018, salesforce.com, inc.

        Example:

            >>> from pl_bolts.models.self_supervised import CoMatch
            ...
             >>> model = CoMatch()

        Train::

            trainer = Trainer()
            trainer.fit(model)

        .. _CoMatch: https://arxiv.org/abs/2011.11183
        """
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
        coefficient_contrastive: float = 1.0,
        contrast_thr: float = 0.8,
        alpha: float = 0.9,
        low_ndembedd: int = 64,
        queue_batch: int = 5,
    ):
        super().__init__(
            ema_eval,
            batch_size,
            mu,
            wresnet_k,
            wresnet_n,
            ema_decay,
            softmax_temperature,
            distribution_alignment,
            coefficient_unsupervised,
            pseudo_thr,
            lr,
            weight_decay,
            momentum,
            gpus,
            max_epochs,
        )
        self.coefficient_contrastive = coefficient_contrastive
        self.contrast_thr = contrast_thr
        self.alpha = alpha
        self.low_ndembedd = low_ndembedd
        self.queue_batch = queue_batch

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.current_step_number = 0

    def on_train_epoch_end(self, unused=None) -> None:
        super().on_train_epoch_end()
        self.current_step_number += 1

    def setup(self, stage):
        super().setup(stage)
        # Init Model
        self.model = WideResnet(n_classes=self.n_classes, k=self.wresnet_k, n=self.wresnet_n, proj=True)
        # Loss
        self.criteria_u = torch.nn.LogSoftmax(dim=1)
        # Mem Bank
        queue_size = self.queue_batch * (self.mu + 1) * self.batch_size
        self.queue_features = Queue(torch.zeros(queue_size, self.low_ndembedd), queue_size)
        self.queue_probs = Queue(torch.zeros(queue_size, self.n_classes), queue_size)

    def _get_similarity_probabilities(self, a, b):
        sim = torch.exp(torch.mm(a, b.t()) / self.softmax_temperature)
        sim = sim / sim.sum(1, keepdim=True)
        return sim

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]  # X
        unlabeled_batch = batch["unlabeled"]  # U

        supervised_imgs, supervised_labels = labeled_batch
        (img_u_weak, img_u_strong0, img_u_strong1), label_u = unlabeled_batch

        batch_size = supervised_imgs.size(0)
        unlabeled_batch_size = img_u_weak.size(0)

        # Concate Different Input together.
        images = self.interleave(
            torch.cat([supervised_imgs, img_u_weak, img_u_strong0, img_u_strong1]), 3 * self.mu + 1
        )
        logits, features = self.model(images)
        logits = self.de_interleave(logits, 3 * self.mu + 1)
        features = self.de_interleave(features, 3 * self.mu + 1)
        # Split logits
        supervised_logits = logits[:batch_size]
        # logits_u_weak, logits_u_strong0, logits_u_strong1 = torch.split(logits[batch_size:], unlabeled_batch_size)
        logits_u_weak, logits_u_strong0, logits_u_strong1 = logits[batch_size:].chunk(3)
        # Split features
        features_x = features[:batch_size]
        # features_u_weak, features_u_strong0, features_u_strong1 = torch.split(
        #     features[batch_size:], unlabeled_batch_size
        # )
        features_u_weak, features_u_strong0, features_u_strong1 = features[batch_size:].chunk(3)

        supervised_loss = self.criteria_x(supervised_logits, supervised_labels)
        with torch.no_grad():
            probs = self.get_unlabled_logits_weak_probs(logits_u_weak)
            mask, label_u_guess = self.get_pesudo_mask_and_infer_u_label(probs)
            probs_orig = probs.clone()

            # Memory Smoothing
            if self.current_epoch > 0 or self.current_step_number > self.queue_batch:
                probs = self.alpha * probs + (1 - self.alpha) * torch.mm(
                    self._get_similarity_probabilities(features_u_weak, self.queue_features()), self.queue_probs()
                )
            features_weak = torch.cat([features_u_weak, features_x], dim=0)
            probs_weak = torch.cat(
                [
                    probs_orig,
                    torch.zeros(batch_size, self.n_classes)
                    .to(self.device)
                    .scatter(1, supervised_labels.view(-1, 1), 1),
                ],
                dim=0,
            )
            # Update memory bank.
            total_batch_size = batch_size + unlabeled_batch_size
            self.queue_features.update(features_weak, total_batch_size)
            self.queue_probs.update(probs_weak, total_batch_size)

        # Embedding Similarity
        sim_probs = self._get_similarity_probabilities(features_u_strong0, features_u_strong1)
        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= self.contrast_thr).float()
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        # contrastive loss
        contrastive_loss = -(torch.log(sim_probs + 1e-7) * Q).sum(1)
        contrastive_loss = contrastive_loss.mean()
        # unsupervised classification loss
        unsupervised_loss = -torch.sum((self.criteria_u(logits_u_strong0) * probs), dim=1) * mask
        unsupervised_loss = unsupervised_loss.mean()

        loss = (
            supervised_loss
            + self.coefficient_unsupervised * unsupervised_loss
            + self.coefficient_contrastive * contrastive_loss
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/supervised_loss", supervised_loss, on_step=True, on_epoch=True)
        self.log("train/unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=True)
        corr_u_label = (label_u_guess == label_u).float() * mask
        self.log("acc@unlabeled", corr_u_label.sum().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc@strong", mask.sum().item(), on_step=True, on_epoch=True)
        self.log("train/mask", mask.mean().item(), on_step=True, on_epoch=True)
        self.log("train/pos_mask", pos_mask.sum(1).float().mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def eval_forward(self, images):
        logits, _ = self.model(images)
        return logits


class CoMatchCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Link args.
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("trainer.gpus", "model.gpus")
        parser.link_arguments("data.mu", "model.mu")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")
        parser.set_defaults({"data.mode": "comatch"})


if __name__ == "__main__":
    from pl_bolts.models.self_supervised.fixmatch.datasets import SSLDataModule

    cli = CoMatchCLI(CoMatch, SSLDataModule)
