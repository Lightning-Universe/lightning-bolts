import warnings

import packaging.version as pv
import pytest
import torch
import torch.nn as nn
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised.swav.swav_swin import swin_b, swin_s, swin_v2_b, swin_v2_s, swin_v2_t
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.transforms.self_supervised.swav_transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
from pl_bolts.utils import _IS_WINDOWS
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning


def check_compatibility():
    return pv.parse(torchvision.__version__) >= pv.parse("0.13")


model = [swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b]


@pytest.mark.parametrize(
    ("model_architecture", "hidden_mlp", "prj_head_type", "feat_dim"),
    [
        (swin_s, 0, nn.Linear, 128),
        (swin_s, 2048, nn.Sequential, 128),
        (swin_b, 0, nn.Linear, 128),
        (swin_b, 2048, nn.Sequential, 128),
        (swin_v2_t, 0, nn.Linear, 128),
        (swin_v2_t, 2048, nn.Sequential, 128),
        (swin_v2_s, 0, nn.Linear, 128),
        (swin_v2_s, 2048, nn.Sequential, 128),
        (swin_v2_b, 0, nn.Linear, 128),
        (swin_v2_b, 2048, nn.Sequential, 128),
    ],
)
@pytest.mark.skipif(not check_compatibility(), reason="Torchvision version not compatible, must be >= 0.13")
@torch.no_grad()
def test_swin_projection_head(model_architecture, hidden_mlp, prj_head_type, feat_dim):
    model = model_architecture(hidden_mlp=hidden_mlp, output_dim=feat_dim)
    assert isinstance(model.projection_head, prj_head_type)


@pytest.mark.parametrize("model", ["swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b"])
@pytest.mark.skipif(not check_compatibility(), reason="Torchvision version not compatible, must be >= 0.13")
@pytest.mark.skipif(_IS_WINDOWS, reason="numpy.core._exceptions._ArrayMemoryError...")  # todo
def test_swav_swin_model(tmpdir, datadir, model, catch_warnings):
    """Test SWAV on CIFAR-10."""
    warnings.filterwarnings(
        "ignore",
        message=".+does not have many workers which may be a bottleneck.+",
        category=PossibleUserWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning)

    batch_size = 2
    datamodule = CIFAR10DataModule(data_dir=datadir, batch_size=batch_size, num_workers=0)

    datamodule.train_transforms = SwAVTrainDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], num_crops=[2, 1], gaussian_blur=False
    )
    datamodule.val_transforms = SwAVEvalDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], num_crops=[2, 1], gaussian_blur=False
    )
    if torch.cuda.device_count() >= 1:
        devices = torch.cuda.device_count()
        accelerator = "gpu"
    else:
        devices = None
        accelerator = "cpu"

    model = SwAV(
        arch=model,
        hidden_mlp=512,
        nodes=1,
        gpus=0 if devices is None else devices,
        num_samples=datamodule.num_samples,
        batch_size=batch_size,
        num_crops=[2, 1],
        sinkhorn_iterations=1,
        num_prototypes=2,
        queue_length=0,
        maxpool1=False,
        first_conv=False,
        dataset="cifar10",
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=True,
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        max_epochs=1,
        logger=True,
    )
    trainer.fit(model, datamodule=datamodule)
