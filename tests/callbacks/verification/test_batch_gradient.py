from unittest.mock import Mock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn

from pl_bolts.callbacks import BatchGradientVerificationCallback
from pl_bolts.callbacks.verification.batch_gradient import default_input_mapping, default_output_mapping, selective_eval
from pl_bolts.utils import BatchGradientVerification
from tests import _MARK_REQUIRE_GPU


class TemplateModel(nn.Module):
    def __init__(self, mix_data=False):
        """Base model for testing.

        The setting ``mix_data=True`` simulates a wrong implementation.
        """
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.bn = nn.BatchNorm1d(10)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        # x: (B, 5, 2)
        if self.mix_data:
            x = x.view(10, -1).permute(1, 0).view(-1, 10)  # oops!
        else:
            x = x.view(-1, 10)  # good!
        return self.linear(self.bn(x))


class MultipleInputModel(TemplateModel):
    """Base model for testing verification when forward accepts multiple arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = (torch.rand(10, 5, 2), torch.rand(10, 5, 2))

    def forward(self, x, y, some_kwarg=True):
        out = super().forward(x) + super().forward(y)
        return out


class MultipleOutputModel(TemplateModel):
    """Base model for testing verification when forward has multiple outputs."""

    def forward(self, x):
        out = super().forward(x)
        return None, out, out, False


class DictInputDictOutputModel(TemplateModel):
    """Base model for testing verification when forward has a collection of outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = {
            "w": 42,
            "x": {"a": torch.rand(3, 5, 2)},
            "y": torch.rand(3, 1, 5, 2),
            "z": torch.tensor(2),
        }

    def forward(self, y, x, z, w):
        out1 = super().forward(x["a"])
        out2 = super().forward(y)
        out3 = out1 + out2
        out = {1: out1, 2: out2, 3: [out1, out3]}
        return out


class LitModel(LightningModule):
    """Base model for testing verification with LightningModules."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DictInputDictOutputModel(*args, **kwargs)
        self.example_input_array = self.model.input_array

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@pytest.mark.parametrize(
    "model_class",
    [TemplateModel, MultipleInputModel, MultipleOutputModel, DictInputDictOutputModel],
)
@pytest.mark.parametrize("mix_data", [True, False])
@pytest.mark.parametrize(
    "device",
    [torch.device("cpu"), pytest.param(torch.device("cuda", 0), marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU))],
)
def test_batch_gradient_verification(model_class, mix_data, device):
    """Test detection of batch gradient mixing with different PyTorch models."""
    model = model_class(mix_data).to(device)
    is_valid = not mix_data
    verification = BatchGradientVerification(model)
    result = verification.check(input_array=model.input_array)
    assert result == is_valid


@pytest.mark.parametrize("mix_data", [True, False])
@pytest.mark.parametrize(
    "device",
    [torch.device("cpu"), pytest.param(torch.device("cuda", 0), marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU))],
)
def test_batch_gradient_verification_pl_module(mix_data, device):
    """Test detection of batch gradient mixing with a LightningModule."""
    model = LitModel(mix_data).to(device)
    is_valid = not mix_data
    verification = BatchGradientVerification(model)
    result = verification.check(input_array=None)
    assert result == is_valid


@pytest.mark.parametrize(
    "gpus",
    [0, pytest.param(1, marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU))],
)
def test_batch_gradient_verification_callback(gpus):
    """Test detection of batch gradient mixing with the callback implementation."""
    trainer = Trainer(gpus=gpus)
    model = LitModel(mix_data=True)

    expected = "Your model is mixing data across the batch dimension."

    callback = BatchGradientVerificationCallback()
    with pytest.warns(UserWarning, match=expected):
        callback.on_train_start(trainer, model)

    callback = BatchGradientVerificationCallback(error=True)
    with pytest.raises(RuntimeError, match=expected):
        callback.on_train_start(trainer, model)


def test_batch_verification_raises_on_batch_size_1():
    """Test that batch gradient verification only works with batch size greater than one."""
    model = TemplateModel()
    verification = BatchGradientVerification(model)
    small_batch = model.input_array[0:1]
    with pytest.raises(MisconfigurationException, match="Batch size must be greater than 1"):
        verification.check(input_array=small_batch)


def test_batch_verification_calls_custom_input_output_mappings():
    """Test that batch gradient verification can support different input and outputs with user-provided
    mappings."""
    model = MultipleInputModel()

    def input_mapping(inputs):
        assert isinstance(inputs, tuple) and len(inputs) == 2
        return [inputs[0]]

    def output_mapping(outputs):
        assert isinstance(outputs, Tensor)
        return torch.cat((outputs, outputs), 1)

    mocked_input_mapping = Mock(wraps=input_mapping)
    mocked_output_mapping = Mock(wraps=output_mapping)
    verification = BatchGradientVerification(model)
    verification.check(
        model.input_array,
        input_mapping=mocked_input_mapping,
        output_mapping=mocked_output_mapping,
    )
    mocked_input_mapping.assert_called_once()
    mocked_output_mapping.assert_called_once()


def test_default_input_mapping():
    """Test the data types and nesting the default input mapping can handle."""
    b = 3
    tensor0 = torch.rand(b, 2, 5)
    tensor1 = torch.rand(b, 9)
    tensor2 = torch.rand(b, 5, 1)

    # Tensor
    data = tensor0.double()
    output = default_input_mapping(data)
    assert len(output) == 1
    assert output[0] is data

    # tuple
    data = ("foo", tensor1, tensor2, [])
    out1, out2 = default_input_mapping(data)
    assert out1 is tensor1
    assert out2 is tensor2

    # dict + nesting
    data = {
        "one": ["foo", tensor2],
        "two": tensor0,
    }
    out2, out0 = default_input_mapping(data)
    assert out2 is tensor2
    assert out0 is tensor0


def test_default_output_mapping():
    """Test the data types and nesting the default output mapping can handle."""
    b = 3
    tensor0 = torch.rand(b, 2, 5)
    tensor1 = torch.rand(b, 9)
    tensor2 = torch.rand(b, 5, 1)
    tensor3 = torch.rand(b)
    scalar = torch.tensor(3.14)

    # Tensor
    data = tensor0.double()
    output = default_output_mapping(data)
    assert output is data

    # tuple + nesting
    data = (tensor0, None, tensor1, "foo", [tensor2])
    expected = torch.cat((tensor0.view(b, -1), tensor1.view(b, -1), tensor2.view(b, -1)), dim=1)
    output = default_output_mapping(data)
    assert torch.all(output == expected)

    # dict + nesting
    data = {
        "one": tensor1,
        "two": {"three": tensor3.double()},  # will convert to float
        "four": scalar,  # ignored
        "five": [tensor0, tensor0],
    }
    expected = torch.cat(
        (
            tensor1.view(b, -1),
            tensor3.view(b, -1),
            tensor0.view(b, -1),
            tensor0.view(b, -1),
        ),
        dim=1,
    )
    output = default_output_mapping(data)
    assert torch.all(output == expected)


class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm0 = nn.BatchNorm1d(2)
        self.batch_norm1 = nn.BatchNorm1d(3)
        self.instance_norm = nn.InstanceNorm1d(4)


def test_selective_eval():
    """Test that the selective_eval context manager only applies to selected layer types."""
    model = BatchNormModel()
    model.train()
    with selective_eval(model, [nn.BatchNorm1d]):
        assert not model.batch_norm0.training
        assert not model.batch_norm1.training
        assert model.instance_norm.training

    assert model.batch_norm0.training
    assert model.batch_norm1.training
    assert model.instance_norm.training


def test_selective_eval_invariant():
    """Test that the selective_eval context manager does not undo layers that were already in eval mode."""
    model = BatchNormModel()
    model.train()
    model.batch_norm1.eval()
    assert model.batch_norm0.training
    assert not model.batch_norm1.training

    with selective_eval(model, [nn.BatchNorm1d]):
        assert not model.batch_norm0.training
        assert not model.batch_norm1.training

    assert model.batch_norm0.training
    assert not model.batch_norm1.training
