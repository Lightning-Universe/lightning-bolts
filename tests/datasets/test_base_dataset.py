import numpy as np
import pytest
import torch
from pl_bolts.datasets.base_dataset import DataModel
from pl_bolts.datasets.utils import to_tensor
from pl_bolts.utils import _IS_WINDOWS


class TestDataModel:
    @pytest.fixture()
    def data(self):
        return np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    def test_process_transform_is_none(self, data):
        dm = DataModel(data=data)
        np.testing.assert_array_equal(dm.process(data[0]), data[0])
        np.testing.assert_array_equal(dm.process(data[1]), data[1])

    @pytest.mark.skipif(  # todo
        _IS_WINDOWS, reason="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64"
    )
    def test_process_transform_is_not_none(self, data):
        dm = DataModel(data=data, transform=to_tensor)
        torch.testing.assert_close(dm.process(data[0]), torch.tensor([1, 0, 0, 1]))
        torch.testing.assert_close(dm.process(data[1]), torch.tensor([0, 1, 1, 0]))
