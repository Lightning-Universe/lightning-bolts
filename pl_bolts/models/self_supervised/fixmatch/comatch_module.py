import torch

from .fixmatch_module import FixMatch
from .networks import WideResnet, get_ema_model


class CoMatch(FixMatch):
    def setup(self, stage):
        if stage == "fit":
            train_loader = self.train_dataloader()
            n_classes = len(train_loader["labeled"].dataset.classes)
            self.model = WideResnet(n_classes=n_classes, k=self.hparams.wresnet_k, n=self.hparams.wresnet_n, proj=True)
            if self.ema_eval:
                self.ema_model = get_ema_model(self.model)
            self.total_steps = (
                len(train_loader["labeled"].dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus))
            ) * float(self.hparams.max_epochs)
            # Mem Bank

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]  # X
        unlabeled_batch = batch["unlabeled"]  # U

        img_x_weak, label_x = labeled_batch
        (img_u_weak, img_u_strong0, img_u_strong1), label_u = unlabeled_batch

        batch_size = img_x_weak.size(0)
        unlabeled_batch_size = img_u_weak.size(0)
        # Concate Different Input together.
        images = torch.cat([img_x_weak, img_u_weak, img_u_strong0, img_u_strong1], dim=0)

        logits, features = self.model(images)
        # Split logits
        logits_x = logits[:batch_size]

        logits_u_weak, logits_u_strong0, logits_u_strong1 = torch.split(logits[batch_size:], unlabeled_batch_size)
        # Split features
        features_x = features[:batch_size]

        features_u_weak, features_u_strong0, features_u_strong1 = torch.split(
            features[batch_size:], unlabeled_batch_size
        )
        loss_x = self.criteria_x(logits_x, label_x)
        with torch.no_grad():
            probs = self.get_unlabled_logits_weak_probs(logits_u_weak)
            scores, label_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(self.hparams.pseudo_thr).float()
            probs_orig = probs.clone()

        loss_u = (self.criteria_u(logits_u_strong0, label_u_guess) * mask).mean()

        loss = loss_x + self.hparams.coefficient_u * loss_u
        self.log("loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("loss_x", loss_x, on_step=True, on_epoch=True, logger=True)
        self.log("loss_u", loss_u, on_step=True, on_epoch=True)
        corr_u_label = (label_u_guess == label_u).float() * mask
        self.log("num of acc@unlabeled", corr_u_label.sum().item(), on_step=True, on_epoch=True)
        self.log("num of strong aug", mask.sum().item(), on_step=True, on_epoch=True)
        self.log("num of mask", mask.mean().item(), on_step=True, on_epoch=True)
        return loss
