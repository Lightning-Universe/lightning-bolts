from typing import Sequence, Union

import numpy as np
import torch

TArrays = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence["TArrays"]]  # type: ignore
