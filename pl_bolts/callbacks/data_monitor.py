from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch import nn as nn
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from pl_bolts.utils import _WANDB_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _WANDB_AVAILABLE:
    import wandb
else:  # pragma: no cover
    warn_missing_pkg("wandb")
    wandb = None  # type: ignore


class DataMonitorBase(Callback):

    supported_loggers = (
        TensorBoardLogger,
        WandbLogger,
    )

    def __init__(self, log_every_n_steps: int = None):
        """
        Base class for monitoring data histograms in a LightningModule.
        This requires a logger configured in the Trainer, otherwise no data is logged.
        The specific class that inherits from this base defines what data gets collected.

        Args:
            log_every_n_steps: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.
        """
        super().__init__()
        self._log_every_n_steps: Optional[int] = log_every_n_steps
        self._log = False
        self._trainer: Trainer
        self._train_batch_idx: int

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log = self._is_logger_available(trainer.logger)
        self._log_every_n_steps = self._log_every_n_steps or trainer.log_every_n_steps  # type: ignore[attr-defined]
        self._trainer = trainer

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._train_batch_idx = batch_idx

    def log_histograms(self, batch: Any, group: str = "") -> None:
        """
        Logs the histograms at the interval defined by `row_log_interval`, given a logger is available.

        Args:
            batch: torch or numpy arrays, or a collection of it (tuple, list, dict, ...), can be nested.
                If the data appears in a dictionary, the keys are used as labels for the corresponding histogram.
                Otherwise the histograms get labelled with an integer index.
                Each label also has the tensors's shape as suffix.
            group: Name under which the histograms will be grouped.
        """
        if not self._log or (self._train_batch_idx + 1) % self._log_every_n_steps != 0:  # type: ignore[operator]
            return

        batch = apply_to_collection(batch, dtype=np.ndarray, function=torch.from_numpy)
        named_tensors: Dict[str, Tensor] = {}
        collect_and_name_tensors(batch, output=named_tensors, parent_name=group)

        for name, tensor in named_tensors.items():
            self.log_histogram(tensor, name)

    def log_histogram(self, tensor: Tensor, name: str) -> None:
        """
        Override this method to customize the logging of histograms.
        Detaches the tensor from the graph and moves it to the CPU for logging.

        Args:
            tensor: The tensor for which to log a histogram
            name: The name of the tensor as determined by the callback. Example: ``ìnput/0/[64, 1, 28, 28]``
        """
        logger = self._trainer.logger
        tensor = tensor.detach().cpu()
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_histogram(tag=name, values=tensor, global_step=self._trainer.global_step)

        if isinstance(logger, WandbLogger):
            if not _WANDB_AVAILABLE:  # pragma: no cover
                raise ModuleNotFoundError("You want to use `wandb` which is not installed yet.")

            logger.experiment.log(data={name: wandb.Histogram(tensor)}, commit=False)

    def _is_logger_available(self, logger: LightningLoggerBase) -> bool:
        available = True
        if not logger:
            rank_zero_warn("Cannot log histograms because Trainer has no logger.")
            available = False
        if not isinstance(logger, self.supported_loggers):
            rank_zero_warn(
                f"{self.__class__.__name__} does not support logging with {logger.__class__.__name__}."
                f" Supported loggers are: {', '.join(map(lambda x: str(x.__name__), self.supported_loggers))}"
            )
            available = False
        return available


class ModuleDataMonitor(DataMonitorBase):

    GROUP_NAME_INPUT = "input"
    GROUP_NAME_OUTPUT = "output"

    def __init__(
        self,
        submodules: Optional[Union[bool, List[str]]] = None,
        log_every_n_steps: int = None,
    ):
        """
        Args:
            submodules: If `True`, logs the in- and output histograms of every submodule in the
                LightningModule, including the root module itself.
                This parameter can also take a list of names of specifc submodules (see example below).
                Default: `None`, logs only the in- and output of the root module.
            log_every_n_steps: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.

        Note:
            A too low value for `log_every_n_steps` may have a significant performance impact
            especially when many submodules are involved, since the logging occurs during the forward pass.
            It should only be used for debugging purposes.

        Example:

            .. code-block:: python

                # log the in- and output histograms of the `forward` in LightningModule
                trainer = Trainer(callbacks=[ModuleDataMonitor()])

                # all submodules in LightningModule
                trainer = Trainer(callbacks=[ModuleDataMonitor(submodules=True)])

                # specific submodules
                trainer = Trainer(callbacks=[ModuleDataMonitor(submodules=["generator", "generator.conv1"])])

        """
        super().__init__(log_every_n_steps=log_every_n_steps)
        self._submodule_names = submodules
        self._hook_handles: List = []

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        submodule_dict = dict(pl_module.named_modules())
        self._hook_handles = []
        for name in self._get_submodule_names(pl_module):
            if name not in submodule_dict:
                rank_zero_warn(
                    f"{name} is not a valid identifier for a submodule in {pl_module.__class__.__name__},"
                    " skipping this key."
                )
                continue
            handle = self._register_hook(name, submodule_dict[name])
            self._hook_handles.append(handle)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for handle in self._hook_handles:
            handle.remove()

    def _get_submodule_names(self, root_module: nn.Module) -> List[str]:
        # default is the root module only
        names = [""]

        if isinstance(self._submodule_names, list):
            names = self._submodule_names

        if self._submodule_names is True:
            names = [name for name, _ in root_module.named_modules()]

        return names

    def _register_hook(self, module_name: str, module: nn.Module) -> RemovableHandle:
        input_group_name = (f"{self.GROUP_NAME_INPUT}/{module_name}" if module_name else self.GROUP_NAME_INPUT)
        output_group_name = (f"{self.GROUP_NAME_OUTPUT}/{module_name}" if module_name else self.GROUP_NAME_OUTPUT)

        def hook(_: Module, inp: Sequence, out: Sequence) -> None:
            inp = inp[0] if len(inp) == 1 else inp
            self.log_histograms(inp, group=input_group_name)
            self.log_histograms(out, group=output_group_name)

        handle = module.register_forward_hook(hook)
        return handle


class TrainingDataMonitor(DataMonitorBase):

    GROUP_NAME = "training_step"

    def __init__(self, log_every_n_steps: int = None):
        """
        Callback that logs the histogram of values in the batched data passed to `training_step`.

        Args:
            log_every_n_steps: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.

        Example:

            .. code-block:: python

                # log histogram of training data passed to `LightningModule.training_step`
                trainer = Trainer(callbacks=[TrainingDataMonitor()])
        """
        super().__init__(log_every_n_steps=log_every_n_steps)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, *args, **kwargs)
        self.log_histograms(batch, group=self.GROUP_NAME)


def collect_and_name_tensors(data: Any, output: Dict[str, Tensor], parent_name: str = "input") -> None:
    """
    Recursively fetches all tensors in a (nested) collection of data (depth-first search) and names them.
    Data in dictionaries get named by their corresponding keys and otherwise they get indexed by an
    increasing integer. The shape of the tensor gets appended to the name as well.

    Args:
        data: A collection of data (potentially nested).
        output: A dictionary in which the outputs will be stored.
        parent_name: Used when called recursively on a nested input data.

    Example:
        >>> data = {"x": torch.zeros(2, 3), "y": {"z": torch.zeros(5)}, "w": 1}
        >>> output = {}
        >>> collect_and_name_tensors(data, output)
        >>> output  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {'input/x/[2, 3]': ..., 'input/y/z/[5]': ...}
    """
    assert isinstance(output, dict)
    if isinstance(data, Tensor):
        name = f"{parent_name}/{shape2str(data)}"
        output[name] = data

    if isinstance(data, dict):
        for k, v in data.items():
            collect_and_name_tensors(v, output, parent_name=f"{parent_name}/{k}")

    if isinstance(data, Sequence) and not isinstance(data, str):
        for i, item in enumerate(data):
            collect_and_name_tensors(item, output, parent_name=f"{parent_name}/{i:d}")


def shape2str(tensor: Tensor) -> str:
    """
    Returns the shape of a tensor in bracket notation as a string.

    Example:
        >>> shape2str(torch.rand(1, 2, 3))
        '[1, 2, 3]'
        >>> shape2str(torch.rand(4))
        '[4]'
    """
    return "[" + ", ".join(map(str, tensor.shape)) + "]"
