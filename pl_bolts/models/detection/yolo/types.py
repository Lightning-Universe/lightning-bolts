from typing import Any, Dict, List, Tuple, Union

from torch import Tensor

IMAGES = Union[Tuple[Tensor, ...], List[Tensor]]
PRED = Dict[str, Any]
PREDS = Union[Tuple[PRED, ...], List[PRED]]
TARGET = Dict[str, Any]
TARGETS = Union[Tuple[TARGET, ...], List[TARGET]]
BATCH = Tuple[IMAGES, TARGETS]
NETWORK_OUTPUT = Tuple[List[Tensor], List[Tensor], List[int]]  # detections, losses, hits
