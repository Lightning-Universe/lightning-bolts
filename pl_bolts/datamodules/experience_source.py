"""
Datamodules for RL models that rely on experiences generated during training

Based on implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
"""
from collections import namedtuple
from typing import Iterable, Callable

from torch.utils.data import IterableDataset

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceSourceDataset(IterableDataset):
    """
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator
