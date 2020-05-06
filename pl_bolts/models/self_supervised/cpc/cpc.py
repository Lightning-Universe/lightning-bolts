import torch
import torch.optim as optim
from torchvision.datasets import STL10, CIFAR10, CIFAR100, SVHN, ImageNet
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from pl_bolts.models.self_supervised.losses import CPCV1LossNCE
import math


class CPCV1(pl.LightningModule):

    # ------------------------------
    # INIT
    # ------------------------------
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # encoder network (Z vectors)
        dummy_batch = torch.zeros((2, 3, hparams.patch_size, hparams.patch_size))
        self.encoder = CPCResNet101(dummy_batch)

        # context network (C vectors)
        c, h = self.__compute_final_nb_c(hparams.patch_size)
        self.context_network = MaskedConv2d(c)

        # W transforms (k > 0)
        self.W_list = {}
        for k in range(1, h):
            w = torch.nn.Linear(c, c)
            self.W_list[str(k)] = w

        self.W_list = torch.nn.ModuleDict(self.W_list)

        # loss (has cached sampling layers, no params)
        self.nce_loss = CPCV1LossNCE()

        self.tng_split = None
        self.val_split = None

    def __compute_final_nb_c(self, patch_size):
        dummy_batch = torch.zeros((2*49, 3, patch_size, patch_size))
        dummy_batch = self.encoder(dummy_batch)
        dummy_batch = self.__recover_z_shape(dummy_batch, 2)
        b, c, h, w = dummy_batch.size()
        return c, h

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)

        return Z

    # ------------------------------
    # FWD
    # ------------------------------
    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        b, p, c, w, h = img_1.size()
        img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)
        Z = self.__recover_z_shape(Z, b)

        return Z

    def training_step(self, batch, batch_nb):
        img_1, _ = batch

        # Latent features
        Z = self.forward(img_1.half())
        Z = Z.half()

        # generate the context vars
        C = self.context_network(Z)

        # apply masked context network

        # ------------------
        # NCE LOSS
        loss = self.nce_loss(Z, C, self.W_list)
        unsupervised_loss = loss
        if self.trainer.use_amp:
            unsupervised_loss = unsupervised_loss.half()

        # ------------------
        # FULL LOSS
        total_loss = unsupervised_loss
        result = {
            'loss': total_loss
        }

        return result

    def validation_step(self, batch, batch_nb):
        img_1, labels = batch

        if self.trainer.use_amp:
            img_1 = img_1.half()

        # generate features
        # Latent features
        Z = self.forward(img_1)
        Z = Z.half()

        # generate the context vars
        C = self.context_network(Z.half())

        # NCE LOSS
        loss = self.nce_loss(Z, C, self.W_list)
        unsupervised_loss = loss

        result = {
            'val_nce': unsupervised_loss
        }
        return result

    def validation_end(self, outputs):
        val_nce = 0
        for output in outputs:
            val_nce += output['val_nce']

        val_nce = val_nce / len(outputs)
        return {'val_nce': val_nce}

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i):
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.8, 0.999),
            weight_decay=1e-5,
            eps=1e-7
        )

        if self.hparams.dataset_name == 'CIFAR10': # Dataset.C100, Dataset.STL10
            lr_scheduler = MultiStepLR(opt, milestones=[250, 280], gamma=0.2)
        else:
            lr_scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.2)

        return [opt], [lr_scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = amdim_utils.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.cifar10_root, train=True, transform=train_transform, download=True)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

        if self.hparams.dataset_name == 'stl_10':
            train_transform = amdim_utils.CPCTransformsSTL10Patches(patch_size=self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = STL10(root=self.hparams.stl10_data_files, split='unlabeled', transform=train_transform, download=True)

            self.tng_split, self.val_split = random_split(dataset, [95000, 5000])

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(self.tng_split)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = amdim_utils.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.imagenet_data_files_tng,
                                        nb_classes=self.hparams.nb_classes,
                                        split='train',
                                        transform=train_transform)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset_name == 'CIFAR10':
            train_transform = amdim_utils.CPCTransformsC10()
            dataset = CIFAR10(root=self.hparams.cifar10_root, train=False, transform=train_transform, download=True)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

        if self.hparams.dataset_name == 'stl_10':
            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(self.val_split)

            loader = DataLoader(
                dataset=self.val_split,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

        if self.hparams.dataset_name == 'imagenet_128':
            train_transform = amdim_utils.CPCTransformsImageNet128Patches(self.hparams.patch_size, overlap=self.hparams.patch_overlap)
            dataset = UnlabeledImagenet(self.hparams.imagenet_data_files_val,
                                        nb_classes=self.hparams.nb_classes,
                                        split='val',
                                        transform=train_transform)

            dist_sampler = None
            if self.trainer.use_ddp:
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=16,
                sampler=dist_sampler
            )

            return loader

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        parser.set_defaults(nb_hopt_trials=1000)
        parser.set_defaults(min_nb_epochs=1000)
        parser.set_defaults(max_nb_epochs=1100)
        parser.set_defaults(early_stop_metric='val_nce')
        parser.set_defaults(model_save_monitor_value='val_nce')
        parser.set_defaults(model_save_monitor_mode='min')
        parser.set_defaults(early_stop_mode='min')

        # CIFAR 10
        patch_size = 8
        cifar_10 = {
            'dataset_name': 'CIFAR10',
            'depth': 10,
            'patch_size': patch_size,
            'batch_size': 200,
            'nb_classes': 10,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                #     2e-4 * (1 / 16), 2e-4 * (1 / 8),
                # 2e-4 * (1 / 4), 2e-4 * (1 / 2),
                2e-5,
                # 2e-4 * 2, 2e-4 * 4,
                #     2e-4 * 8, 2e-4 * 16
            ]
        }

        # stl-10
        patch_size = 16
        stl_10 = {
            'dataset_name': 'stl_10',
            'depth': 8,
            'patch_size': patch_size,
            'batch_size': 200,
            'nb_classes': 10,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                # 2e-7,
                # 2e-4*(1/4),
                # 2e-4*(1/2),
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
                # 2e-4*2,
                # 3e-4,
                # 2e-4*3,
                # 2e-4*4,
                # 8e-4,
                # 2e-4 * 8, 2e-4 * 16
            ]
        }

        # imagenet_128
        patch_size = 32
        imagenet_128 = {
            'dataset_name': 'imagenet_128',
            'depth': 10,
            'patch_size': patch_size,
            'batch_size': 60,
            'nb_classes': 1000,
            'patch_overlap': patch_size // 2,
            'lr_options': [
                2e-6,
                2e-5,
                2e-4,
                2e-3,
                2e-2
            ]
        }

        # dataset = cifar_10
        # dataset = stl_10
        dataset = imagenet_128

        # dataset options
        parser.opt_list('--nb_classes', default=dataset['nb_classes'], type=int, options=[10], tunable=False)
        parser.opt_list('--patch_size', default=dataset['patch_size'], type=int, options=[10], tunable=False)
        parser.opt_list('--patch_overlap', default=dataset['patch_overlap'], type=int, options=[10], tunable=False)

        # trainin params
        resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
        parser.add_argument('--dataset_name', type=str, default=dataset['dataset_name'])
        parser.opt_list('--batch_size', type=int, default=dataset['batch_size'], options=[120, 140], help='input batch size (default: 200)', tunable=False)
        parser.opt_list('--learning_rate', type=float, default=0.0001, options=dataset['lr_options'], tunable=True)

        # data
        parser.opt_list('--voc_root', default=f'{root_dir}/fisherman/datasets', type=str, tunable=False)
        parser.opt_list('--cifar10_root', default=f'{root_dir}/fisherman/datasets', type=str, tunable=False)
        parser.opt_list('--cifar100_root', default=f'{root_dir}/fisherman/datasets', type=str, tunable=False)
        parser.opt_list('--svhn_root', default=f'{root_dir}/fisherman/datasets', type=str, tunable=False)
        parser.opt_list('--stl10_data_files', default=f'{root_dir}/fisherman/datasets/stl10', type=str, tunable=False)
        parser.opt_list('--imagenet_data_files_tng', default=f'{root_dir}/fisherman/datasets/imagenet/train', type=str, tunable=False)
        parser.opt_list('--imagenet_data_files_test', default=f'{root_dir}/fisherman/datasets/imagenet/test', type=str, tunable=False)
        parser.opt_list('--imagenet_data_files_val', default=f'{root_dir}/fisherman/datasets/imagenet/val', type=str, tunable=False)
        parser.opt_list('--imagenet_data_files_debug', default='/Users/someUser/Developer/nyu/fisherman/fisherman/datasets/imagenet/debug', type=str, tunable=False)

        return parser