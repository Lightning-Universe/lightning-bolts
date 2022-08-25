from typing import Sequence, Union

import numpy as np
import torch

TArrays = Union[torch.Tensor, np.ndarray, Sequence[Union[float, int]], Sequence["TArrays"]]
