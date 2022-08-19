import collections.abc as container_abcs
import re
from queue import Queue
from threading import Thread
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch._six import string_classes
from torch.utils.data import DataLoader, Dataset

from pl_bolts.utils.stability import under_review


@under_review()
class AsynchronousLoader:
    """Class for asynchronously loading from CPU memory to device memory with DataLoader.

    Note that this only works for single GPU training, multiGPU uses PyTorch's DataParallel or
    DistributedDataParallel which uses its own code for transferring data across GPUs. This could just
    break or make things slower with DataParallel or DistributedDataParallel.

    Args:
        data: The PyTorch Dataset or DataLoader we're using to load.
        device: The PyTorch device we are loading to
        q_size: Size of the queue used to store the data loaded to the device
        num_batches: Number of batches to load. This must be set if the dataloader
            doesn't have a finite __len__. It will also override DataLoader.__len__
            if set and DataLoader has a __len__. Otherwise it can be left as None
        **kwargs: Any additional arguments to pass to the dataloader if we're
            constructing one here
    """

    def __init__(
        self,
        data: Union[DataLoader, Dataset],
        device: torch.device = torch.device("cuda", 0),
        q_size: int = 10,
        num_batches: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(data, torch.utils.data.DataLoader):
            self.dataloader = data
        else:
            self.dataloader = DataLoader(data, **kwargs)

        if num_batches is not None:
            self.num_batches = num_batches
        elif hasattr(self.dataloader, "__len__"):
            self.num_batches = len(self.dataloader)
        else:
            raise Exception("num_batches must be specified or data must have finite __len__")

        self.device = device
        self.q_size = q_size

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue: Queue = Queue(maxsize=self.q_size)

        self.idx = 0

        self.np_str_obj_array_pattern = re.compile(r"[SaUO]")

    def load_loop(self) -> None:  # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))
            if i == len(self):
                break

    # Recursive loading for each instance based on torch.utils.data.default_collate
    def load_instance(self, sample: Any) -> Any:
        elem_type = type(sample)

        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                # Can only do asynchronous transfer if we use pin_memory
                if not sample.is_pinned():
                    sample = sample.pin_memory()
                return sample.to(self.device, non_blocking=True)
        elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
            if elem_type.__name__ == "ndarray" and self.np_str_obj_array_pattern.search(sample.dtype.str) is not None:
                return self.load_instance(sample)
            return self.load_instance(torch.as_tensor(sample))
        elif isinstance(sample, container_abcs.Mapping):
            return {key: self.load_instance(sample[key]) for key in sample}
        elif isinstance(sample, tuple) and hasattr(sample, "_fields"):  # namedtuple
            return elem_type(*(self.load_instance(d) for d in sample))
        elif isinstance(sample, container_abcs.Sequence) and not isinstance(sample, string_classes):
            return [self.load_instance(s) for s in sample]
        else:
            return sample

    def __iter__(self) -> "AsynchronousLoader":
        # We don't want to run the thread more than once
        # Start a new thread if we are at the beginning of a new epoch, and our current worker is dead

        if_worker = not hasattr(self, "worker") or not self.worker.is_alive()  # type: ignore[has-type]
        if if_worker and self.queue.empty() and self.idx == 0:
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True
            self.worker.start()
        return self

    def __next__(self) -> Tensor:
        # If we've reached the number of batches to return
        # or the queue is empty and the worker is dead then exit
        done = not self.worker.is_alive() and self.queue.empty()
        done = done or self.idx >= len(self)
        if done:
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # Otherwise return the next batch
        out = self.queue.get()
        self.queue.task_done()
        self.idx += 1
        return out

    def __len__(self) -> int:
        return self.num_batches
