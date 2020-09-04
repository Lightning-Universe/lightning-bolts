from unittest.mock import call, MagicMock, patch

import torch

from pl_bolts.loggers import AzureMlLogger


def test_azureml_logger():
    mock_run = MagicMock()
    with patch('azureml.core.Run.get_context', return_value=mock_run):
        logger = AzureMlLogger()
        assert logger.experiment is mock_run
        assert logger.name is mock_run.id
        assert logger.version == str(mock_run.number)


def test_azureml_additional_methods():
    mock_run = MagicMock()
    logger = AzureMlLogger(mock_run)
    assert logger.experiment is mock_run

    logger.log_hyperparams({"a": 1})
    mock_run.tag.assert_called_once_with("a", 1)
    mock_run.reset_mock()

    logger.log_metrics({"torch": torch.ones(1), "float": 2.1})
    mock_run.log.assert_has_calls([call("torch", 1), call("float", 2.1)], any_order=True)
    mock_run.reset_mock()

    logger.log_image("test", "image_file")
    mock_run.log_image.assert_called_once_with("test", "image_file", None, "")
    mock_run.reset_mock()

    logger.log_list("test", [1, 2, 3], "description")
    mock_run.log_list.assert_called_once_with("test", [1, 2, 3], "description")
    mock_run.reset_mock()

    logger.log_table("test", {"a": [1], "b": [2]})
    mock_run.log_table.assert_called_once_with("test", {"a": [1], "b": [2]}, "")
    mock_run.reset_mock()

    logger.finalize("test")
    mock_run.flush.assert_called_once_with()
    mock_run.reset_mock()
