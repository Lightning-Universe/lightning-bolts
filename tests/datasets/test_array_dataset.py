import numpy as np
import pytest
import torch
from pytorch_lightning.utilities import exceptions

from pl_bolts.datasets import ArrayDataset, DataModel
from pl_bolts.datasets.utils import to_tensor


def add_one(integers: np.ndarray) -> np.ndarray:
    output = []
    for data in integers:
        output.append(data + 1)
    return np.array(output)


def add_one_to_tensor(integers: np.ndarray) -> torch.Tensor:
    integers = add_one(integers)
    return to_tensor(integers)


class TestArrayDataset:
    @pytest.fixture
    def array_dataset(self):
        features_1 = DataModel(
            data=[[1, 0, -1, 2], [1, 0, -2, -1], [2, 5, 0, 3], [-7, 1, 2, 2]], transform=add_one_to_tensor
        )
        target_1 = DataModel(data=[1, 0, 0, 1], transform=to_tensor)

        features_2 = DataModel(data=np.array([[2, 1, -5, 1], [1, 0, -2, -1], [2, 5, 0, 3], [-7, 1, 2, 2]]))
        target_2 = DataModel(data=[1, 0, 1, 1])
        return ArrayDataset(features_1, target_1, features_2, target_2)

    def test_len(self, array_dataset):
        assert len(array_dataset) == 4

    def test_getitem_with_transforms(self, array_dataset):
        assert len(array_dataset[0]) == 4
        assert len(array_dataset[1]) == 4
        assert len(array_dataset[2]) == 4
        assert len(array_dataset[3]) == 4
        torch.testing.assert_close(array_dataset[0][0], torch.tensor([2, 1, 0, 3]))
        torch.testing.assert_close(array_dataset[0][1], torch.tensor(1))
        np.testing.assert_array_equal(array_dataset[0][2], np.array([2, 1, -5, 1]))
        assert array_dataset[0][3] == 1
        torch.testing.assert_close(array_dataset[1][0], torch.tensor([2, 1, -1, -0]))
        torch.testing.assert_close(array_dataset[1][1], torch.tensor(0))
        np.testing.assert_array_equal(array_dataset[1][2], np.array([1, 0, -2, -1]))
        assert array_dataset[1][3] == 0
        torch.testing.assert_close(array_dataset[2][0], torch.tensor([3, 6, 1, 4]))
        torch.testing.assert_close(array_dataset[2][1], torch.tensor(0))
        np.testing.assert_array_equal(array_dataset[2][2], np.array([2, 5, 0, 3]))
        assert array_dataset[2][3] == 1
        torch.testing.assert_close(array_dataset[3][0], torch.tensor([-6, 2, 3, 3]))
        torch.testing.assert_close(array_dataset[3][1], torch.tensor(1))
        np.testing.assert_array_equal(array_dataset[3][2], np.array([-7, 1, 2, 2]))
        assert array_dataset[3][3] == 1

    def test__equal_size_true(self, array_dataset):
        assert array_dataset._equal_size() is True

    def test__equal_size_false(self):
        features = DataModel(data=[[1, 0, 1]])
        target = DataModel([1, 0, 1])
        with pytest.raises(exceptions.MisconfigurationException):
            ArrayDataset(features, target)
