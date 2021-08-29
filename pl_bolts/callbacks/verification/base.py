# type: ignore
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Optional

import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import move_data_to_device, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature


class VerificationBase:
    """Base class for model verification.

    All verifications should run with any :class:`torch.nn.Module` unless otherwise stated.
    """

    def __init__(self, model: nn.Module):
        """
        Arguments:
            model: The model to run verification for.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def check(self, *args: Any, **kwargs: Any) -> bool:
        """Runs the actual test on the model. All verification classes must implement this.

        Arguments:
            *args: Any positional arguments that are needed to run the test
            *kwargs: Keyword arguments that are needed to run the test

        Returns:
            `True` if the test passes, and `False` otherwise. Some verifications can only be performed
            with a heuristic accuracy, thus the return value may not always reflect the true state of
            the system in these cases.
        """

    def _get_input_array_copy(self, input_array: Optional[Any] = None) -> Any:
        """Returns a deep copy of the example input array in cases where it is expected that the input changes
        during the verification process.

        Arguments:
            input_array: The input to clone.
        """
        if input_array is None and isinstance(self.model, LightningModule):
            input_array = self.model.example_input_array
        input_array = deepcopy(input_array)

        if isinstance(self.model, LightningModule):
            kwargs = {}
            if is_param_in_hook_signature(self.model.transfer_batch_to_device, "dataloader_idx"):
                # Requires for Lightning 1.4 and above
                kwargs["dataloader_idx"] = 0
            input_array = self.model.transfer_batch_to_device(input_array, self.model.device, **kwargs)
        else:
            input_array = move_data_to_device(input_array, device=next(self.model.parameters()).device)

        return input_array

    def _model_forward(self, input_array: Any) -> Any:
        """Feeds the input array to the model via the ``__call__`` method.

        Arguments:
            input_array: The input that goes into the model. If it is a tuple, it gets
                interpreted as the sequence of positional arguments and is passed in by tuple unpacking.
                If it is a dict, the contents get passed in as named parameters by unpacking the dict.
                Otherwise, the input array gets passed in as a single argument.

        Returns:
            The output of the model.
        """
        if isinstance(input_array, tuple):
            return self.model(*input_array)
        if isinstance(input_array, dict):
            return self.model(**input_array)
        return self.model(input_array)


class VerificationCallbackBase(Callback):
    """Base class for model verification in form of a callback.

    This type of verification is expected to only work with
    :class:`~pytorch_lightning.core.lightning.LightningModule` and will take the input array
    from :attr:`~pytorch_lightning.core.lightning.LightningModule.example_input_array` if needed.
    """

    def __init__(self, warn: bool = True, error: bool = False) -> None:
        """
        Arguments:
            warn: If ``True``, prints a warning message when verification fails. Default: ``True``.
            error: If ``True``, prints an error message when verification fails. Default: ``False``.
        """
        self._raise_warning = warn
        self._raise_error = error

    def message(self, *args: Any, **kwargs: Any) -> str:
        """
        The message to be printed when the model does not pass the verification.
        If the message for warning and error differ, override the
        :meth:`warning_message` and :meth:`error_message`
        methods directly.

        Arguments:
            *args: Any positional arguments that are needed to construct the message.
            **kwargs: Any keyword arguments that are needed to construct the message.

        Returns:
            The message as a string.
        """

    def warning_message(self, *args: Any, **kwargs: Any) -> str:
        """The warning message printed when the model does not pass the verification."""
        return self.message(*args, **kwargs)

    def error_message(self, *args: Any, **kwargs: Any) -> str:
        """The error message printed when the model does not pass the verification."""
        return self.message(*args, **kwargs)

    def _raise(self, *args: Any, **kwargs: Any) -> None:
        if self._raise_error:
            raise RuntimeError(self.error_message(*args, **kwargs))
        if self._raise_warning:
            rank_zero_warn(self.warning_message(*args, **kwargs))
