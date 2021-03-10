from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import move_data_to_device

from pl_bolts.callbacks.verification.base import VerificationBase
from tests import _MARK_REQUIRE_GPU


class TrivialVerification(VerificationBase):

    def check(self, *args, **kwargs):
        return True


class PyTorchModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 2)

    def forward(self, *args):
        return args


class LitModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.example_input_array = None
        self.model = PyTorchModel()

    def forward(self, *args):
        return self.model(*args)


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu"),
     pytest.param(torch.device("cuda", 0), marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU))],
)
def test_verification_base_get_input_array(device):
    """ Test that the base class calls the correct methods to transfer the input to the device the model is on. """
    model = PyTorchModel().to(device)
    verification = TrivialVerification(model)
    input_tensor = torch.rand(5)
    assert verification.model == model

    # for a PyTorch model, user must provide the input array
    with patch("pl_bolts.callbacks.verification.base.move_data_to_device", wraps=move_data_to_device) as mocked:
        copied_tensor = verification._get_input_array_copy(input_array=input_tensor)
        mocked.assert_called_once()
        assert copied_tensor.device == device
        assert torch.allclose(input_tensor, copied_tensor.cpu())

    model = LitModel().to(device)
    model.example_input_array = input_tensor
    verification = TrivialVerification(model)

    # for a LightningModule, user can rely on the example_input_array
    with patch.object(model, "transfer_batch_to_device", wraps=model.transfer_batch_to_device) as mocked:
        copied_tensor = verification._get_input_array_copy(input_array=None)
        mocked.assert_called_once()
        assert copied_tensor.device == model.device == device
        assert torch.allclose(model.example_input_array, copied_tensor.cpu())


def test_verification_base_model_forward_for_input_array():
    """ Test that the input_array is correctly fed to the forward method depending on its type. """
    model = Mock()
    verification = TrivialVerification(model)

    # tuple must be passed as positional args
    input_array = (1, torch.tensor(2), None)
    verification._model_forward(input_array)
    model.assert_called_with(1, torch.tensor(2), None)

    # dict must be passed as keyword args
    input_array = {"one": 1, "two": torch.tensor(2), "three": None}
    verification._model_forward(input_array)
    model.assert_called_with(one=1, two=torch.tensor(2), three=None)

    # everything else will be passed directly
    input_array = torch.rand(2)
    verification._model_forward(input_array)
    model.assert_called_with(input_array)
