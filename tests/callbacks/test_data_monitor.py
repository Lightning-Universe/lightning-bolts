import warnings
from unittest import mock
from unittest.mock import call

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch import nn

from pl_bolts.callbacks import ModuleDataMonitor, TrainingDataMonitor
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models import LitMNIST
from pytorch_lightning import Trainer
from torch import nn


# @pytest.mark.parametrize(("log_every_n_steps", "max_steps", "expected_calls"), [pytest.param(3, 10, 3)])
@mock.patch("pl_bolts.callbacks.data_monitor.DataMonitorBase.log_histogram")
def test_base_log_interval_override(
    log_histogram, tmpdir, datadir, catch_warnings, log_every_n_steps=3, max_steps=10, expected_calls=3
):
    """Test logging interval set by log_every_n_steps argument."""
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )
    monitor = TrainingDataMonitor(log_every_n_steps=log_every_n_steps)
    model = LitMNIST(num_workers=0)
    datamodule = MNISTDataModule(data_dir=datadir)
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        max_steps=max_steps,
        callbacks=[monitor],
        accelerator="auto",
    )

    trainer.fit(model, datamodule=datamodule)
    assert log_histogram.call_count == (expected_calls * 2)  # 2 tensors per log call


@pytest.mark.parametrize(
    ("log_every_n_steps", "max_steps", "expected_calls"),
    [
        pytest.param(1, 5, 5),
        pytest.param(2, 5, 2),
        pytest.param(5, 5, 1),
        pytest.param(6, 5, 0),
    ],
)
@mock.patch("pl_bolts.callbacks.data_monitor.DataMonitorBase.log_histogram")
def test_base_log_interval_fallback(
    log_histogram,
    tmpdir,
    log_every_n_steps,
    max_steps,
    expected_calls,
    datadir,
    catch_warnings,
):
    """Test that if log_every_n_steps not set in the callback, fallback to what is defined in the Trainer."""
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )
    monitor = TrainingDataMonitor()
    model = LitMNIST(num_workers=0)
    datamodule = MNISTDataModule(data_dir=datadir)
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        callbacks=[monitor],
        accelerator="auto",
    )
    trainer.fit(model, datamodule=datamodule)
    assert log_histogram.call_count == (expected_calls * 2)  # 2 tensors per log call


def test_base_no_logger_warning(catch_warnings):
    """Test a warning is displayed when Trainer has no logger."""
    monitor = TrainingDataMonitor()
    trainer = Trainer(logger=False, callbacks=[monitor], accelerator="auto", max_epochs=-1)
    with pytest.warns(UserWarning, match="Cannot log histograms because Trainer has no logger"):
        monitor.on_train_start(trainer, pl_module=LightningModule())


def test_base_unsupported_logger_warning(tmpdir, catch_warnings):
    """Test a warning is displayed when an unsupported logger is used."""
    warnings.filterwarnings(
        "ignore",
        message=".*is deprecated in v1.6.*",
        category=LightningDeprecationWarning,
    )
    monitor = TrainingDataMonitor()
    trainer = Trainer(
        logger=LoggerCollection([TensorBoardLogger(tmpdir)]),
        callbacks=[monitor],
        accelerator="auto",
        max_epochs=1,
    )
    with pytest.warns(UserWarning, match="does not support logging with LoggerCollection"):
        monitor.on_train_start(trainer, pl_module=LightningModule())


@mock.patch("pl_bolts.callbacks.data_monitor.TrainingDataMonitor.log_histogram")
def test_training_data_monitor(log_histogram, tmpdir, datadir, catch_warnings):
    """Test that the TrainingDataMonitor logs histograms of data points going into training_step."""
    monitor = TrainingDataMonitor()
    model = LitMNIST(data_dir=datadir)
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        callbacks=[monitor],
        accelerator="auto",
        max_epochs=1,
    )
    monitor.on_train_start(trainer, model)

    # single tensor
    example_data = torch.rand(2, 3, 4)
    monitor.on_train_batch_start(trainer, model, batch=example_data, batch_idx=0)
    assert log_histogram.call_args_list == [
        call(example_data, "training_step/[2, 3, 4]"),
    ]

    log_histogram.reset_mock()

    # tuple
    example_data = (torch.rand(2, 3, 4), torch.rand(5), "non-tensor")
    monitor.on_train_batch_start(trainer, model, batch=example_data, batch_idx=0)
    assert log_histogram.call_args_list == [
        call(example_data[0], "training_step/0/[2, 3, 4]"),
        call(example_data[1], "training_step/1/[5]"),
    ]

    log_histogram.reset_mock()

    # dict
    example_data = {
        "x0": torch.rand(2, 3, 4),
        "x1": torch.rand(5),
        "non-tensor": "non-tensor",
    }
    monitor.on_train_batch_start(trainer, model, batch=example_data, batch_idx=0)
    assert log_histogram.call_args_list == [
        call(example_data["x0"], "training_step/x0/[2, 3, 4]"),
        call(example_data["x1"], "training_step/x1/[5]"),
    ]


class SubModule(nn.Module):
    def __init__(self, inp, out) -> None:
        super().__init__()
        self.sub_layer = nn.Linear(inp, out)

    def forward(self, *args, **kwargs):
        return self.sub_layer(*args, **kwargs)


class ModuleDataMonitorModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(12, 5)
        self.layer2 = SubModule(5, 2)

    def forward(self, x):
        x = x.flatten(1)
        self.layer1_input = x
        x = self.layer1(x)
        self.layer1_output = x
        x = torch.relu(x + 1)
        self.layer2_input = x
        x = self.layer2(x)
        self.layer2_output = x
        x = torch.relu(x - 2)
        return x


@mock.patch("pl_bolts.callbacks.data_monitor.ModuleDataMonitor.log_histogram")
def test_module_data_monitor_forward(log_histogram, tmpdir, catch_warnings):
    """Test that the default ModuleDataMonitor logs inputs and outputs of model's forward."""
    monitor = ModuleDataMonitor(submodules=None)
    model = ModuleDataMonitorModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        callbacks=[monitor],
        accelerator="auto",
        max_epochs=1,
    )
    monitor.on_train_start(trainer, model)
    monitor.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    example_input = torch.rand(2, 6, 2)
    output = model(example_input)
    assert log_histogram.call_args_list == [
        call(example_input, "input/[2, 6, 2]"),
        call(output, "output/[2, 2]"),
    ]


@mock.patch("pl_bolts.callbacks.data_monitor.ModuleDataMonitor.log_histogram")
def test_module_data_monitor_submodules_all(log_histogram, tmpdir, catch_warnings):
    """Test that the ModuleDataMonitor logs the inputs and outputs of each submodule."""
    monitor = ModuleDataMonitor(submodules=True)
    model = ModuleDataMonitorModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        callbacks=[monitor],
        accelerator="auto",
        max_epochs=1,
    )
    monitor.on_train_start(trainer, model)
    monitor.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    example_input = torch.rand(2, 6, 2)
    output = model(example_input)
    assert log_histogram.call_args_list == [
        call(model.layer1_input, "input/layer1/[2, 12]"),
        call(model.layer1_output, "output/layer1/[2, 5]"),
        call(model.layer2_input, "input/layer2.sub_layer/[2, 5]"),
        call(model.layer2_output, "output/layer2.sub_layer/[2, 2]"),
        call(model.layer2_input, "input/layer2/[2, 5]"),
        call(model.layer2_output, "output/layer2/[2, 2]"),
        call(example_input, "input/[2, 6, 2]"),
        call(output, "output/[2, 2]"),
    ]


@mock.patch("pl_bolts.callbacks.data_monitor.ModuleDataMonitor.log_histogram")
def test_module_data_monitor_submodules_specific(log_histogram, tmpdir, catch_warnings):
    """Test that the ModuleDataMonitor logs the inputs and outputs of selected submodules."""
    monitor = ModuleDataMonitor(submodules=["layer1", "layer2.sub_layer"])
    model = ModuleDataMonitorModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        callbacks=[monitor],
        accelerator="auto",
        max_epochs=1,
    )
    monitor.on_train_start(trainer, model)
    monitor.on_train_batch_start(trainer, model, batch=None, batch_idx=0)

    example_input = torch.rand(2, 6, 2)
    _ = model(example_input)
    assert log_histogram.call_args_list == [
        call(model.layer1_input, "input/layer1/[2, 12]"),
        call(model.layer1_output, "output/layer1/[2, 5]"),
        call(model.layer2_input, "input/layer2.sub_layer/[2, 5]"),
        call(model.layer2_output, "output/layer2.sub_layer/[2, 2]"),
    ]
