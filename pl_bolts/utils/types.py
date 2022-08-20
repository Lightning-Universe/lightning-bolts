from typing import List, Union

import numpy as np
import torch

ARRAYS = Union[torch.Tensor, np.ndarray, List[Union[float, int]], List[List[Union[float, int]]]]
