import os
import pytorch_lightning as pl
from argparse import ArgumentParser

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50, resnet18

from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.models.self_supervised.swav.swav_module import SwAV
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.models.self_supervised.swav.transforms import SwAVFinetuneTransform


def cli_main():  # pragma: no-cover
    from pl_bolts.datamodules import STL10DataModule, ImagenetDataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--dataset', type=str, help='cifar10', default='stl10')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
    parser.add_argument('--data_path', type=str, help='path to ckpt', default=os.getcwd())

    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
    args = parser.parse_args()

    if args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        args.num_samples = 0

        dm.train_transforms = SwAVFinetuneTransform(
            normalize=stl10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=False
        )
        dm.val_transforms = SwAVFinetuneTransform(
            normalize=stl10_normalization(),
            input_height=dm.size()[-1],
            eval_transform=True
        )

        args.maxpool1 = False
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    backbone = SwAV(
        gpus=1,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        datamodule=dm
    ).load_from_checkpoint(args.ckpt_path, strict=False)

    tuner = SSLFineTuner(backbone, in_features=2048, num_classes=dm.num_classes, hidden_dim=None)
    trainer = pl.Trainer.from_argparse_args(
        args, gpus=1, precision=16, early_stop_callback=True
    )
    trainer.fit(tuner, dm)

    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
