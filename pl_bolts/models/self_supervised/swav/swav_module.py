"""
TODO:
- check mlp for eval (verify callback)
- correct stl eval
- LARC check compare
- scheduler not using warmup as optimizer is not
- add swav val data transforms, add val
- check maxpool is false and is actually 1, 1 in the model
- len(train_loader when) DDP with multiple GPUs
- unlabeled batch issue

Adapted from official swav implementation: https://github.com/facebookresearch/swav
"""
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50
from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.models.self_supervised.swav.swav_transforms SwAVTrainDataTransform
from pl_bolts.optimizers.lars_scheduling import LARSWrapper


class SwAV(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        nmb_prototypes: int = 3072,
        first_conv: bool = True,
        maxpool1: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.gpus > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.model = self.init_model()

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, len(train_loader) * args.warmup_epochs
        )

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

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

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
        # use lars wrapper
        pass

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs= False
    ):
        # TODO: warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

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

        parser.add_argument("--nmb_crops", type=int, default=[2, 4], nargs="+",
                            help="list of number of crops (example: [2, 6])")
        parser.add_argument("--size_crops", type=int, default=[96, 36], nargs="+",
                            help="crops resolutions (example: [224, 96])")
        parser.add_argument("--min_scale_crops", type=float, default=[0.33, 0.10], nargs="+",
                            help="argument in RandomResizedCrop (example: [0.14, 0.05])")
        parser.add_argument("--max_scale_crops", type=float, default=[1, 0.33], nargs="+",
                            help="argument in RandomResizedCrop (example: [1., 0.14])")

        # training params
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("-num_workers", default=16, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument('--exclude_bn_bias', default=False, type=bool, help="exclude bn/bias from weight decay")
        parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
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
        parser.add_argument("--queue_length", type=int, default=7680, help="length of the queue (0 for no queue)")
        parser.add_argument("--epoch_queue_starts", type=int, default=15,
                            help="from this epoch, we start using a queue")
        parser.add_argument("--freeze_prototypes_nepochs", default=1, type=int,
                            help="freeze the prototypes during this many epochs from the start")
        

def cli_main():

    if args.dataset == 'stl10':
        dm = STL10DataModule()
        dm.train
        args.maxpool1 = False
    elif:
        raise NotImplementedError("other datasets have not been implemented till now")

    # LR logger callback

    trainer = pl.Trainer(
        gpus=args.gpus,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=16,
        callbacks=[]
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
