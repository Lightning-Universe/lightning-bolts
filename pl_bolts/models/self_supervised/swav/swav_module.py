"""
TODO:
- multi-gpu (need data sampler to adjust length of data_loader)
- correct stl eval
- add swav val data transforms, add val
- len(train_loader when) DDP with multiple GPUs
- unlabeled batch issue

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
from torch.optim import Adam, SGD

from pytorch_lightning.callbacks import LearningRateLogger
from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule

from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.models.self_supervised.swav.swav_transforms import SwAVTrainDataTransform, SwAVEvalDataTransform
from pl_bolts.models.self_supervised.swav.swav_online_eval import SwavOnlineEvaluator
from pl_bolts.optimizers.lars_scheduling import LARSWrapper


class SwAV(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        datamodule: pl.LightningDataModule,
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
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.datamodule = datamodule
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
        global_batch_size = self.gpus * self.batch_size
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
        self.queue_path = os.path.join(self.queue_path, "queue" + str(self.trainer.global_rank) + ".pth")
        if os.path.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)["queue"]

    def init_model(self):
        return resnet50(
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
                ).cuda()

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
        if isinstance(self.datamodule, STL10DataModule):
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        inputs, y = batch
        inputs = inputs[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.module.prototypes.weight.copy_(w)

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
                            self.model.module.prototypes.weight.t()
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
        for param_group in optimizer.optim.param_groups:
            param_group["lr"] = lr_schedule[self.trainer.global_step]

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

            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B

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

            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])

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
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")

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
        parser.add_argument("-num_workers", default=16, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument('--exclude_bn_bias', default=False, type=bool, help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
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
        parser.add_argument("--nmb_prototypes", default=256, type=int, help="number of prototypes")
        parser.add_argument("--queue_length", type=int, default=7680,
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

    # LR logger callback
    lr_logger = LearningRateLogger()

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
        gpus=args.gpus,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=16,
        callbacks=[lr_logger, online_evaluator],
        fast_dev_run=args.fast_dev_run
    )

    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
