import numpy as np
import torch.testing

from pl_bolts.datasets.utils import to_tensor


class TestToTensor:
    def test_to_tensor_list(self):
        _list = [1, 2, 3]
        torch.testing.assert_close(to_tensor(_list), torch.tensor(_list))

    def test_to_tensor_array(self):
        _array = np.array([1, 2, 3])
        torch.testing.assert_close(to_tensor(_array), torch.tensor(_array))

    def test_to_tensor_sequence_(self):
        _sequence = [[1.0, 2.0, 3.0]]
        torch.testing.assert_close(to_tensor(_sequence), torch.tensor(_sequence))
