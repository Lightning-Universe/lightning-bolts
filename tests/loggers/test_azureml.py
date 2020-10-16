from mlflow.tracking.client import MlflowClient
from pl_bolts.loggers import AzureMlLogger
from unittest.mock import MagicMock, patch


def test_azureml_logger():
    mock_run = MagicMock(spec={})
    with patch('azureml.core.Run.get_context', return_value=mock_run):
        logger = AzureMlLogger()
        assert isinstance(logger.experiment, MlflowClient)
        assert logger.name is not None
