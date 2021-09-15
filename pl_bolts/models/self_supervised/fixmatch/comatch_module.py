import torch

from .fixmatch_module import FixMatch
from .networks import WideResnet, get_ema_model


class CoMatch(FixMatch):
    def setup(self, stage):
        super(CoMatch, self).setup(stage)
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

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("-a", "--arch", metavar="ARCH", default="wideresnet")
        parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                            help="mini-batch size (default: 16), this is the total "
                                 "batch size of all GPUs on the current node when "
                                 "using Data Parallel or Distributed Data Parallel",
                            )
        # SSL related args.
        parser.add_argument("--eval-step", type=int, default=1024, help="eval step in Fix Match.")
        parser.add_argument("--expand-labels", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--distribution-alignment", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--pseudo-thr", type=float, default=0.95, help="pseudo label threshold")
        parser.add_argument("--coefficient-u", type=float, default=1.0, help="coefficient of unlabeled loss")
        # Model related args.
        parser.add_argument("--ema-decay", type=float, default=0.999)
        parser.add_argument("--wresnet-k", default=8, type=int, help="width factor of wide resnet")
        parser.add_argument("--wresnet-n", default=28, type=int, help="depth of wide resnet")
        # Training related args.
        parser.add_argument(
            "-lr", "--learning-rate", default=0.03, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument("--wd", "--weight-decay", default=5e-4, type=float, metavar="W",
                            help="weight decay (default: 5e-4)", dest="weight_decay")
        parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use layer0-trained model")
        return parser
