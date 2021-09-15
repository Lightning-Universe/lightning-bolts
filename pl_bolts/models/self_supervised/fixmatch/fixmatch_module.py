from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer

from .lr_scheduler import WarmupCosineLrScheduler
from .networks import WideResnet, ema_model_update, get_ema_model


class FixMatch(LightningModule):
    def __init__(self, ema_eval=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ema_eval = ema_eval

        self.criteria_x = nn.CrossEntropyLoss()
        self.criteria_u = nn.CrossEntropyLoss(reduction="none")
        self.prob_list = []

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        if stage == "fit":
            train_loader = self.train_dataloader()
            self.n_classes = len(train_loader["labeled"].dataset.classes)
            self.model = WideResnet(n_classes=self.n_classes, k=self.hparams.wresnet_k, n=self.hparams.wresnet_n)
            if self.ema_eval:
                self.ema_model = get_ema_model(self.model)
            self.total_steps = (
                len(train_loader["labeled"].dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus))
            ) * float(self.hparams.max_epochs)

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]  # X
        unlabeled_batch = batch["unlabeled"]  # U

        img_x_weak, label_x = labeled_batch
        (img_u_weak, img_u_strong), label_u = unlabeled_batch

        batch_size = img_u_weak.size(0)
        mu = int(img_u_weak.size(0) // batch_size)
        imgs = torch.cat([img_x_weak, img_u_weak, img_u_strong], dim=0)
        logits = self.model(imgs)
        logits_x = logits[:batch_size]
        logits_u_weak, logits_u_strong = torch.split(logits[batch_size:], batch_size * mu)
        supervised_loss = self.criteria_x(logits_x, label_x)
        with torch.no_grad():
            probs = self.get_unlabled_logits_weak_probs(logits_u_weak)
            mask, label_u_guess = self.get_pesudo_mask_and_infer_u_label(probs)

        unsupervised_loss = (self.criteria_u(logits_u_strong, label_u_guess) * mask).mean()

        loss = supervised_loss + self.hparams.coefficient_unsupervised * unsupervised_loss
        self.log("loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=True, logger=True)
        self.log("unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=True)
        corr_u_label = (label_u_guess == label_u).float() * mask
        self.log("num of acc@unlabeled", corr_u_label.sum().item(), on_step=True, on_epoch=True)
        self.log("num of strong aug", mask.sum().item(), on_step=True, on_epoch=True)
        self.log("num of mask", mask.mean().item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        loss = self.criteria_x(logits, labels)
        acc1, acc5 = self.__accuracy(logits, labels, topk=(1, 5))
        # ema eval
        if self.ema_eval:
            with torch.no_grad():
                ema_model_update(self.model, self.ema_model, self.hparams.ema_decay)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )
        scheduler = WarmupCosineLrScheduler(optimizer, max_iter=self.total_steps, warmup_iter=0)
        return [optimizer], [scheduler]

    def get_unlabled_logits_weak_probs(self, logits_u_weak):
        probs = torch.softmax(logits_u_weak, dim=1)
        if self.hparams.distribution_alignment:
            self.prob_list.append(probs.mean(0))
            if len(self.prob_list) > 32:
                self.prob_list.pop(0)
            prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def get_pesudo_mask_and_infer_u_label(self, probs):
        scores, label_u_guess = torch.max(probs, dim=1)
        mask = scores.ge(self.hparams.pseudo_thr).float()
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
        parser.add_argument('--mu', default=7, type=int,
                            help='coefficient of unlabeled batch size')
        parser.add_argument("--num-labeled", type=int, default=4000, help="number of labeled samples for training")
        parser.add_argument("--eval-step", type=int, default=1024, help="eval step in Fix Match.")
        parser.add_argument("--expand-labels", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--distribution_alignment", action="store_true", help="expand labels in SSL.")
        parser.add_argument("--pseudo-thr", type=float, default=0.95, help="pseudo label threshold")
        parser.add_argument("--coefficient-unsupervised", type=float, default=1.0, help="coefficient of unlabeled loss")
        # Model related args.
        parser.add_argument("--ema-decay", type=float, default=0.999)
        parser.add_argument("--wresnet-k", default=8, type=int, help="width factor of wide resnet")
        parser.add_argument("--wresnet-n", default=28, type=int, help="depth of wide resnet")
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


def cli_main():
    from pl_bolts.models.self_supervised.fixmatch.datasets import SSLDataModule

    parent_parser = ArgumentParser(add_help=False)
    parent_parser = Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument("--data-path", metavar="DIR", type=str, help="path to dataset", default="./data")
    parent_parser.add_argument(
        "--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "stl10", "imagenet"]
    )
    parent_parser.add_argument(
        "-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set"
    )
    parent_parser.add_argument("--seed", type=int, default=42, help="seed for initializing training.")
    parser = FixMatch.add_model_specific_args(parent_parser)
    parser.set_defaults(deterministic=True, max_epochs=300)
    args = parser.parse_args()
    dm = SSLDataModule(
        args.data_path,
        args.dataset,
        mu=args.mu,
        num_labeled=args.num_labeled,
        batch_size=args.batch_size,
        eval_step=args.eval_step,
        expand_labels=args.expand_labels,
    )
    model = FixMatch(**vars(args))
    trainer = Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    cli_main()
