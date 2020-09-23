"""
Adapted from official swav implementation: https://github.com/facebookresearch/swav
"""
from argparse import ArgumentParser

import os
import math
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.optim import Adam, SGD

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50, resnet18
from warnings import warn

try:
    from pl_bolts.datamodules import STL10DataModule, ImagenetDataModule
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True

from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.models.self_supervised.swav.transforms import SwAVTrainDataTransform, SwAVEvalDataTransform
from pl_bolts.models.self_supervised.swav.swav_online_eval import SwavOnlineEvaluator
from pl_bolts.optimizers.lars_scheduling import LARSWrapper


class SwAV(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str = 'stl10',
        arch: str = 'resnet50',
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3072,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        queue_length: int = 0,  # must be divisible by total batch-size
        queue_path: str = "queue",
        epoch_queue_starts: int = 15,
        crops_for_assign: list = [0, 1],
        nmb_crops: list = [2, 6],
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = 'adam',
        lars_wrapper: bool = True,
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optimizer: optimizer to use
            lars_wrapper: use LARS wrapper over the optimizer
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.queue_length = queue_length
        self.queue_path = queue_path
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        if self.gpus > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.model = self.init_model()

        # compute iters per epoch
        global_batch_size = self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([self.final_lr + 0.5 * (self.learning_rate - self.final_lr) * (
            1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs)))
        ) for t in iters])

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        self.queue = None
        self.softmax = nn.Softmax(dim=1)

    def setup(self, stage):
        queue_folder = os.path.join(self.logger.log_dir, self.queue_path)
        if not os.path.exists(queue_folder):
            os.makedirs(queue_folder)

        self.queue_path = os.path.join(
            queue_folder,
            "queue" + str(self.trainer.global_rank) + ".pth"
        )

        if os.path.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)["queue"]

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
        elif self.arch == 'resnet50':
            backbone = resnet50

        return backbone(
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1
        )

    def forward(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

                if self.gpus > 0:
                    self.queue = self.queue.cuda()

        self.use_the_queue = False

    def on_train_epoch_end(self) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        if self.dataset == 'stl10':
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        inputs, y = batch
        inputs = inputs[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # 4. time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, on_step=True, on_epoch=False)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_step=False, on_epoch=True)
        return result

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(),
                weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        learning_rate = {'learning_rate': self.lr_schedule[self.trainer.global_step]}
        self.logger.log_metrics(learning_rate, step=self.trainer.global_step)

        # from lightning implementation
        if using_native_amp:
            self.trainer.scaler.step(optimizer)
        else:
            optimizer.step()

    def sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action='store_true')
        parser.add_argument("--fp32", action='store_true')

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="stl10", help="stl10")
        parser.add_argument("--data_path", type=str, default=".", help="path to download data")
        parser.add_argument("--queue_path", type=str, default="queue", help="path for queue")

        parser.add_argument("--nmb_crops", type=int, default=[2, 4], nargs="+",
                            help="list of number of crops (example: [2, 6])")
        parser.add_argument("--size_crops", type=int, default=[96, 36], nargs="+",
                            help="crops resolutions (example: [224, 96])")
        parser.add_argument("--min_scale_crops", type=float, default=[0.33, 0.10], nargs="+",
                            help="argument in RandomResizedCrop (example: [0.14, 0.05])")
        parser.add_argument("--max_scale_crops", type=float, default=[1, 0.33], nargs="+",
                            help="argument in RandomResizedCrop (example: [1., 0.14])")

        # training params
        parser.add_argument("--fast_dev_run", action='store_true')
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        # swav params
        parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                            help="list of crops id used for computing assignments")
        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--epsilon", default=0.05, type=float,
                            help="regularization parameter for Sinkhorn-Knopp algorithm")
        parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                            help="number of iterations in Sinkhorn-Knopp algorithm")
        parser.add_argument("--nmb_prototypes", default=512, type=int, help="number of prototypes")
        parser.add_argument("--queue_length", type=int, default=0,
                            help="length of the queue (0 for no queue); must be divisible by total batch size")
        parser.add_argument("--epoch_queue_starts", type=int, default=15,
                            help="from this epoch, we start using a queue")
        parser.add_argument("--freeze_prototypes_epochs", default=1, type=int,
                            help="freeze the prototypes during this many epochs from the start")

        return parser


def cli_main():
    parser = ArgumentParser()

    # model args
    parser = SwAV.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        dm.train_transforms = SwAVTrainDataTransform(
            normalize=stl10_normalization(),
            size_crops=args.size_crops,
            nmb_crops=args.nmb_crops,
            min_scale_crops=args.min_scale_crops,
            max_scale_crops=args.max_scale_crops,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength
        )

        dm.val_transforms = SwAVEvalDataTransform(
            normalize=stl10_normalization(),
            size_crops=args.size_crops,
            nmb_crops=args.nmb_crops,
            min_scale_crops=args.min_scale_crops,
            max_scale_crops=args.max_scale_crops,
            gaussian_blur=args.gaussian_blur,
            jitter_strength=args.jitter_strength
        )

        args.maxpool1 = False
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    # swav model init
    model = SwAV(**args.__dict__, datamodule=dm)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SwavOnlineEvaluator(
            drop_p=0.,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=[online_evaluator] if args.online_ft else None,
        fast_dev_run=args.fast_dev_run
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
