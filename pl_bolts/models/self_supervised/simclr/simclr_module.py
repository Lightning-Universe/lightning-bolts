from argparse import ArgumentParser

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision.models import densenet
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover

from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.models.self_supervised.resnets import resnet50_bn
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    # TODO:
    def __init__(self,
                 batch_size: int,
                 num_samples: int,
                 warmup_epochs: int = 10,
                 lr: float = 1e-4,
                 opt_weight_decay: float = 1e-6,
                 loss_temperature: float = 0.5,
                 **kwargs):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()

        # h -> || -> z
        self.projection = Projection()

    # TODO:
    def init_encoder(self):
        encoder = resnet50_bn(return_all_feature_maps=False)

        # when using cifar10, replace the first conv so image doesn't shrink away
        encoder.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        return encoder

    # TODO:
    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    # TODO:
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    # TODO:
    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

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
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        if self.lars_wrapper:
            for param_group in optimizer.optim.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        # from lightning
        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, imagenet2012, stl10')

        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--optimizer', choices=['adam', 'lars'], default='lars')
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--lars_momentum', type=float, default=0.9)
        parser.add_argument('--lars_eta', type=float, default=0.001)
        parser.add_argument('--lr_sched_step', type=float, default=30, help='lr scheduler step')
        parser.add_argument('--lr_sched_gamma', type=float, default=0.5, help='lr scheduler step')
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        # Model
        parser.add_argument('--loss_temperature', type=float, default=0.5)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
    from pl_bolts.datamodules import STL10DataModule, CIFAR10DataModule, ImagenetDataModule

    parser = ArgumentParser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

########################
# TODO: chagen prams and transforms for simclr
    if args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False

        normalization = stl10_normalization()
    elif args.dataset == 'cifar10':
        args.batch_size = 2
        args.num_workers = 0

        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False

        normalization = cifar10_normalization()

        # cifar10 specific params
        args.size_crops = [32, 16]
        args.nmb_crops = [2, 1]
        args.gaussian_blur = False
    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.size_crops = [224, 96]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.batch_size = 64
        args.nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = 'sgd'
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3

        args.nmb_prototypes = 3000
        args.online_ft = True

        dm = ImagenetDataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )
########################

    model = SimCLR(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
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
        num_nodes=args.nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=[online_evaluator] if args.online_ft else None,
        fast_dev_run=args.fast_dev_run
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
