from argparse import ArgumentParser

import torch

from .fixmatch_module import FixMatch
from .networks import WideResnet


class Queue:
    def __init__(self, value, size):
        self.value = value
        self.size = size
        self.ptr = 0

    def __call__(self):
        return self.value

    def update(self, new_value, total_batch_size):
        self.value = self.value[self.ptr: self.ptr + total_batch_size, :] = new_value
        self.ptr = (self.ptr + total_batch_size) % self.size


class CoMatch(FixMatch):
    def setup(self, stage):
        super().setup(stage)
        # Init Model
        self.model = WideResnet(n_classes=self.n_classes, k=self.hparams.wresnet_k, n=self.hparams.wresnet_n, proj=True)
        # Loss
        self.criteria_u = torch.nn.LogSoftmax(dim=1)
        # Mem Bank
        queue_size = self.hparams.queue_batch * (self.hparams.mu + 1) * self.hparams.batch_size
        self.queue_features = Queue(torch.zeros(self.queue_size, self.hparams.low_ndembedd).to(self.device), queue_size)
        self.queue_probs = Queue(torch.zeros(self.queue_size, self.n_classes).to(self.device), queue_size)

    def train_epoch_start(self):
        self.current_step_number = 0

    def _get_similarity_probabilities(self, a, b):
        sim = torch.exp(torch.mm(a, b.t()) / self.hparams.temperature)
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
        images = torch.cat([supervised_imgs, img_u_weak, img_u_strong0, img_u_strong1], dim=0)

        logits, features = self.model(images)
        # Split logits
        supervised_logits = logits[:batch_size]
        logits_u_weak, logits_u_strong0, logits_u_strong1 = torch.split(logits[batch_size:], unlabeled_batch_size)
        # Split features
        features_x = features[:batch_size]
        features_u_weak, features_u_strong0, features_u_strong1 = torch.split(
            features[batch_size:], unlabeled_batch_size
        )

        supervised_loss = self.criteria_x(supervised_logits, supervised_labels)
        with torch.no_grad():
            probs = self.get_unlabled_logits_weak_probs(logits_u_weak)
            mask, label_u_guess = self.get_pesudo_mask_and_infer_u_label(probs)
            probs_orig = probs.clone()

            # Memory Smoothing
            if self.current_epoch > 0 or self.current_step_number > self.hparams.queue_batch:
                probs = self.hparams.alpha * probs + (1 - self.hparams.alpha) * torch.mm(
                    self._get_similarity_probabilities(features_u_weak, self.queue_features().t()), self.queue_probs()
                )

            features_weak = torch.cat([features_u_weak, features_x], dim=0)
            probs_weak = torch.cat(
                [probs_orig, torch.zeros(batch_size, self.n_classes).scatter(1, supervised_labels.view(-1, 1), 1).to(self.device)]
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
        pos_mask = (Q >= self.hparams.contrast_thr).float()
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
                + self.hparams.coefficient_unsupervised * unsupervised_loss
                + self.hparams.coefficient_contrastive * contrastive_loss
        )
        self.log("loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=True, logger=True)
        self.log("unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=True)
        corr_u_label = (label_u_guess == label_u).float() * mask
        self.log("num_acc@unlabeled", corr_u_label.sum().item(), on_step=True, on_epoch=True)
        self.log("num_strong_aug", mask.sum().item(), on_step=True, on_epoch=True)
        self.log("num_mask", mask.mean().item(), on_step=True, on_epoch=True)
        self.log("num_pos", pos_mask.sum(1).float().mean().item(), on_step=True, on_epoch=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("-a", "--arch", metavar="ARCH", default="wideresnet")
        parser.add_argument(
            "-b",
            "--batch-size",
            default=16,
            type=int,
            metavar="N",
            help="mini-batch size (default: 16), this is the total "
                 "batch size of all GPUs on the current node when "
                 "using Data Parallel or Distributed Data Parallel",
        )
        # SSL related args.
        parser.add_argument("--eval-step", type=int, default=1024, help="eval step in Fix Match.")
        parser.add_argument("--expand-labels", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--distribution-alignment", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--pseudo-thr", type=float, default=0.95, help="pseudo label threshold")
        parser.add_argument("--coefficient-unsupervised", type=float, default=1.0, help="coefficient of unlabeled loss")
        parser.add_argument(
            "--coefficient-contrastive", type=float, default=1.0, help="coefficient of contrastive loss"
        )
        parser.add_argument("--contrast-thr", default=0.8, type=float, help="pseudo label graph threshold")
        # Model related args.
        parser.add_argument("--alpha", type=float, default=0.9)
        parser.add_argument("--mu", type=int, default=7, help="factor of train batch size of unlabeled samples")
        parser.add_argument("--temperature", default=0.2, type=float, help="softmax temperature")
        parser.add_argument("--low-ndembedd", type=int, default=64, help="Dimension of low dimension embedding.")
        parser.add_argument("--ema-decay", type=float, default=0.999)
        parser.add_argument("--wresnet-k", default=8, type=int, help="width factor of wide resnet")
        parser.add_argument("--wresnet-n", default=28, type=int, help="depth of wide resnet")
        parser.add_argument("--queue-batch", type=float, default=5, help="number of batches stored in memory bank")
        # Training related args.
        parser.add_argument(
            "-lr", "--learning-rate", default=0.03, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=5e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 5e-4)",
            dest="weight_decay",
        )
        parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use layer0-trained model")
        return parser
