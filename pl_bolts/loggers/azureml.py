# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Azure Machine Learning
----------------------
"""

from argparse import Namespace
from typing import BinaryIO, Optional, Dict, Union, Any

try:
    from azureml.core import Run as AzureMlRun
    from azureml.core.run import _OfflineRun as AzureMlOfflineRun
    import matplotlib.pyplot as PyPlot
except ImportError:  # pragma: no-cover
    AzureMlOfflineRun = None
    AzureMlRun = None
    PyPlot = None
    _AZURE_ML_AVAILABLE = False
else:
    _AZURE_ML_AVAILABLE = True


import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_only


class AzureMlLogger(LightningLoggerBase):
    r"""
    Log using `Azure Machine Learning <https://docs.microsoft.com/en-us/azure/machine-learning/>`_. Install it with pip:
    .. code-block:: bash
        pip install azureml-sdk
    The Azure Machine Learning logger will log to standard output if running in
    offline mode, or to
    `Azure Machine Learning metrics <https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments>`_
    if running remotely.
    Example:
        >>> from azureml.core import Run
        >>> from pytorch_lightning.loggers import AzureMlLogger
        >>> run = Run.get_context()
        >>> azureml_logger = AzureMlLogger(run)
        >>> trainer = Trainer(max_epochs=10, logger=azureml_logger)
    Args:
        run: Optionally inject Azure Machine Learning ``Run`` object directly.
            If this is not provided, default to ``Run.get_context()``.
    """

    def __init__(self, run: Optional[AzureMlRun] = None):

        if not _AZURE_ML_AVAILABLE:
            raise ImportError(
                "You want to use `azureml-sdk` logger which is not installed yet,"
                " install it with `pip install azureml-sdk`."
            )
        super().__init__()

        if run is None:
            self._experiment = AzureMlRun.get_context(allow_offline=True)
        else:
            self._experiment = run

    @property
    @rank_zero_experiment
    def experiment(self) -> AzureMlRun:
        r"""
        Actual Azure Machine Learning `Run` object. To use Azure ML features in
        your :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_azure_ml_function()
        """
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Tags the underlying Azure Machine Learning run with the given
        hyperparameters.
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for name, value in params.items():
            self.experiment.tag(name, value)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # Azure ML expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()
            self.experiment.log(key, val)

    @rank_zero_only
    def log_image(self, name: str, path: Union[str, BinaryIO] = None, plot: PyPlot = None, description: str = ""):
        """
        Log an image metric to the run record.
        """
        self.experiment.log_image(name, path, plot, description)

    @rank_zero_only
    def log_list(self, name: str, value: list, description: str = ""):
        """
        Log a list of metric values to the run with the given name.
        """
        self.experiment.log_list(name, value, description)

    @rank_zero_only
    def log_table(self, name: str, value: dict, description: str = ""):
        """
        Log a table metric to the run with the given name. Expects a dictionary
        mapping column name to value.
        """
        self.experiment.log_table(name, value, description)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """
        Calls `.flush()` on the underlying Azure Machine Learning `Run`
        object. This ensures all logs have been sent to Azure Machine Learning.
        """
        self.experiment.flush()

    @property
    def save_dir(self) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        """
        Returns the ID of the underlying Azure Machine Learning `Run` object.
        """
        return self.experiment.id

    @property
    def version(self) -> str:
        r"""
        Returns the run number of the underlying Azure Machine Learning `Run`
        object. Defaults to `0` if the run is offline.
        """
        if isinstance(self.experiment, AzureMlOfflineRun):
            return "0"
        else:
            return str(self.experiment.number)
