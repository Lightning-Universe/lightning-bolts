from unittest.mock import patch, Mock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn as nn

from pl_bolts.callbacks import BatchGradientVerificationCallback
from pl_bolts.callbacks.verification import BatchGradientVerification
from pl_bolts.callbacks.verification.batch_gradient import default_input_mapping, default_output_mapping


class TemplateModel(nn.Module):
    def __init__(self, mix_data=False):
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        # x: (B, 5, 2)
        if self.mix_data:
            x = x.view(10, -1).permute(1, 0).view(-1, 10)  # oops!
        else:
            x = x.view(-1, 10)  # good!
        return self.linear(x)


class MultipleInputModel(TemplateModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = (torch.rand(10, 5, 2), torch.rand(10, 5, 2))

    def forward(self, x, y, some_kwarg=True):
        out = super().forward(x) + super().forward(y)
        return out


class MultipleOutputModel(TemplateModel):
    def forward(self, x):
        out = super().forward(x)
        return None, out, out, False


class DictInputDictOutputModel(TemplateModel):
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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DictInputDictOutputModel(*args, **kwargs)
        self.example_input_array = self.model.input_array

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@pytest.mark.parametrize(
    "model_class",
    [TemplateModel, MultipleInputModel, MultipleOutputModel, DictInputDictOutputModel,],
)
@pytest.mark.parametrize("mix_data", [True, False])
@pytest.mark.parametrize("device", [
    pytest.param(torch.device("cpu")),
    pytest.param(
        torch.device("cuda", 0),
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU"),
    )
])
def test_batch_gradient_verification(model_class, mix_data, device):
    model = model_class(mix_data).to(device)
    is_valid = not mix_data
    verification = BatchGradientVerification(model)
    assert verification.model == model
    result = verification.check(input_array=model.input_array)
    assert result == is_valid


@pytest.mark.parametrize("gpus", [
    pytest.param(0),
    pytest.param(1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")),
])
def test_batch_gradient_verification_callback(gpus):
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
    model = TemplateModel()
    verification = BatchGradientVerification(model)
    small_batch = model.input_array[0:1]
    with pytest.raises(MisconfigurationException, match="Batch size must be greater than 1"):
        verification.check(input_array=small_batch)


def test_batch_verification_calls_custom_input_output_mappings():
    model = MultipleInputModel()

    def input_mapping(inputs):
        assert isinstance(inputs, tuple) and len(inputs) == 2
        return [inputs[0]]

    def output_mapping(outputs):
        assert isinstance(outputs, torch.Tensor)
        return torch.cat((outputs, outputs), 1)

    mocked_input_mapping = Mock(wraps=input_mapping)
    mocked_output_mapping = Mock(wraps=output_mapping)
    verification = BatchGradientVerification(model)
    verification.check(
        model.input_array,
        input_mapping=mocked_input_mapping,
        output_mapping=mocked_output_mapping
    )
    mocked_input_mapping.assert_called_once()
    mocked_output_mapping.assert_called_once()


def test_default_input_mapping():
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
    expected = torch.cat(
        (tensor0.view(b, -1), tensor1.view(b, -1), tensor2.view(b, -1)), dim=1
    )
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
