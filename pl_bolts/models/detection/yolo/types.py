from typing import Any, Dict, List, Tuple

from torch import Tensor

TARGET = Dict[str, Any]
TARGETS = List[TARGET]
NETWORK_OUTPUT = Tuple[List[Tensor], List[Tensor], List[int]]  # detections, losses, hits
