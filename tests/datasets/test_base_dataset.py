import numpy as np
import pytest

from pl_bolts.datasets.base_dataset import DataModel


def add_two(integers: np.ndarray) -> np.ndarray:
    output = []
    for data in integers:
        output.append(data + 2)
    return np.array(output)


class TestDataModel:
    @pytest.fixture
    def data(self):
        return np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    def test_process_transform_is_none(self, data):
        dm = DataModel(data=data)
        np.testing.assert_array_equal(dm.process(data[0]), data[0])
        np.testing.assert_array_equal(dm.process(data[1]), data[1])

    def test_process_transform_is_not_none(self, data):
        dm = DataModel(data=data, transform=add_two)
        np.testing.assert_array_equal(dm.process(data[0]), np.array([3, 2, 2, 3]))
        np.testing.assert_array_equal(dm.process(data[1]), np.array([2, 3, 3, 2]))
