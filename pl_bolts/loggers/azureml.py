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


try:
    from azureml.core import Run as AzureMlRun
except ImportError:  # pragma: no-cover
    AzureMlOfflineRun = None
    _AZURE_ML_AVAILABLE = False
else:
    _AZURE_ML_AVAILABLE = True


from pytorch_lightning.loggers import MLFlowLogger
from typing import Optional
import uuid


class AzureMlLogger(MLFlowLogger):
    r"""
    Log using `Azure Machine Learning <https://docs.microsoft.com/en-us/azure/machine-learning/>`_.
    Install it with pip:

    .. code-block:: bash

        pip install azureml-mlflow

    The Azure Machine Learning logger will log to standard output if running in
    offline mode, or to
    `Azure Machine Learning metrics <https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments>`_
    via
    `MLFlow <https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow>`_
    if running remotely.

    **Online and offline mode**

    .. code-block:: python

        from azureml.core import Run
        from pytorch_lightning.loggers import AzureMlLogger

        # Optional: this is the default value if no run argument is provided.
        run = Run.get_context()
        azureml_logger = AzureMlLogger(run)
        trainer = Trainer(max_epochs=10, logger=azureml_logger)

    Args:
        run: Optionally inject Azure Machine Learning `Run` object directly.
            If this is not provided, default to `Run.get_context()`.
    """

    def __init__(self, run: Optional[AzureMlRun] = None):

        if not _AZURE_ML_AVAILABLE:
            raise ImportError(
                "You want to use `azureml-sdk` logger which is not installed yet,"
                " install it with `pip install azureml-sdk`."
            )

        if run is None:
            run = AzureMlRun.get_context(allow_offline=True)
        
        try:
            experiment = run.experiment
            tracking_uri = experiment.workspace.get_mlflow_tracking_uri()
            experiment_name = experiment.name
        except AttributeError:
            tracking_uri = None
            experiment_name = str(uuid.uuid4())

        super().__init__(experiment_name, tracking_uri)
