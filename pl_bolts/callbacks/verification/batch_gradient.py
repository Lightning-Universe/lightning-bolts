# type: ignore
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, Type

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from pl_bolts.callbacks.verification.base import VerificationBase, VerificationCallbackBase
from pl_bolts.utils.stability import under_review


@under_review()
class BatchGradientVerification(VerificationBase):
    """Checks if a model mixes data across the batch dimension.

    This can happen if reshape- and/or permutation operations are carried out in the wrong order or on the wrong tensor
    dimensions.
    """

    NORM_LAYER_CLASSES = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
    )

    def check(
        self,
        input_array: Any,
        input_mapping: Optional[Callable] = None,
        output_mapping: Optional[Callable] = None,
        sample_idx: int = 0,
    ) -> bool:
        """Runs the test for data mixing across the batch.

        Arguments:
            input_array: A dummy input for the model. Can be a tuple or dict in case the model takes
                multiple positional or named arguments.
            input_mapping: An optional input mapping that returns all batched tensors in a input collection.
                By default, we handle nested collections (tuples, lists, dicts) of tensors and pull them
                out. If your batch is a custom object, you need to provide this input mapping yourself.
                See :func:`default_input_mapping` for more information on the default behavior.
            output_mapping: An optional output mapping that combines all batched tensors in the output
                collection into one big batch of shape (B, N), where N is the total number of dimensions
                that follow the batch dimension in each tensor. By default, we handle nested collections
                (tuples, lists, dicts) of tensors and combine them automatically. See
                :func:`default_output_mapping` for more information on the default behavior.
            sample_idx:
                The index `i` of the batch sample to run the test for. When computing the gradient of
                a loss value on the `i-th` output w.r.t. the whole input, we expect the gradient to be
                non-zero only on the `i-th` input sample and zero gradient on the rest of the batch.

        Returns:
             ``True`` if the data in the batch does not mix during the forward pass, and ``False`` otherwise.
        """
        input_mapping = input_mapping or default_input_mapping
        output_mapping = output_mapping or default_output_mapping
        input_array = self._get_input_array_copy(input_array)
        input_batches = input_mapping(input_array)

        if input_batches[0].size(0) < 2:
            raise MisconfigurationException("Batch size must be greater than 1 to run verification.")

        for input_batch in input_batches:
            input_batch.requires_grad = True

        self.model.zero_grad()
        with selective_eval(self.model, self.NORM_LAYER_CLASSES):
            output = self._model_forward(input_array)

        # backward on the i-th sample should lead to gradient only in i-th input slice
        output_mapping(output)[sample_idx].sum().backward()

        zero_grad_inds = list(range(len(input_batches[0])))
        zero_grad_inds.pop(sample_idx)

        has_grad_outside_sample = [input_batch.grad[zero_grad_inds].abs().sum().item() for input_batch in input_batches]
        has_grad_inside_sample = [input_batch.grad[sample_idx].abs().sum().item() for input_batch in input_batches]
        return not any(has_grad_outside_sample) and all(has_grad_inside_sample)


@under_review()
class BatchGradientVerificationCallback(VerificationCallbackBase):
    """The callback version of the :class:`BatchGradientVerification` test.

    Verification is performed right before training begins.
    """

    def __init__(
        self,
        input_mapping: Optional[Callable] = None,
        output_mapping: Optional[Callable] = None,
        sample_idx: int = 0,
        **kwargs: Any,
    ):
        """
        Arguments:
            input_mapping: An optional input mapping that returns all batched tensors in a input collection.
                See :meth:`BatchGradientVerification.check` for more information.
            output_mapping: An optional output mapping that combines all batched tensors in the output
                collection into one big batch. See :meth:`BatchGradientVerification.check` for more information.
            sample_idx: The index of the batch sample to run the test for.
                See :meth:`BatchGradientVerification.check` for more information.
            **kwargs: Additional arguments for the base class :class:`VerificationCallbackBase`
        """
        super().__init__(**kwargs)
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._sample_idx = sample_idx

    def message(self, *args: Any, **kwargs: Any) -> str:
        message = (
            "Your model is mixing data across the batch dimension."
            " This can lead to wrong gradient updates in the optimizer."
            " Check the operations that reshape and permute tensor dimensions in your model."
        )
        return message

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        verification = BatchGradientVerification(pl_module)
        result = verification.check(
            input_array=pl_module.example_input_array,
            input_mapping=self._input_mapping,
            output_mapping=self._output_mapping,
            sample_idx=self._sample_idx,
        )
        if not result:
            self._raise()


@under_review()
def default_input_mapping(data: Any) -> List[Tensor]:
    """Finds all tensors in a (nested) collection that have the same batch size.

    Args:
        data: a tensor or a collection of tensors (tuple, list, dict, etc.).

    Returns:
        A list of all tensors with the same batch dimensions. If the input was already a tensor, a one-
        element list with the tensor is returned.

    >>> data = (torch.zeros(3, 1), "foo", torch.ones(3, 2), torch.rand(2))
    >>> result = default_input_mapping(data)
    >>> len(result)
    2
    >>> result[0].shape
    torch.Size([3, 1])
    >>> result[1].shape
    torch.Size([3, 2])
    """
    tensors = collect_tensors(data)
    batches: List[Tensor] = []
    for tensor in tensors:
        if tensor.ndim > 0 and (not batches or tensor.size(0) == batches[0].size(0)):
            batches.append(tensor)
    return batches


@under_review()
def default_output_mapping(data: Any) -> Tensor:
    """Pulls out all tensors in a output collection and combines them into one big batch for verification.

    Args:
        data: a tensor or a (nested) collection of tensors (tuple, list, dict, etc.).

    Returns:
        A float tensor with shape (B, N) where B is the batch size and N is the sum of (flattened)
        dimensions of all tensors in the collection. If the input was already a tensor, the tensor
        itself is returned.

    Example:
        >>> data = (torch.rand(3, 5), "foo", torch.rand(3, 2, 4))
        >>> result = default_output_mapping(data)
        >>> result.shape
        torch.Size([3, 13])
        >>> data = {"one": torch.rand(3, 5), "two": torch.rand(3, 2, 1)}
        >>> result = default_output_mapping(data)
        >>> result.shape
        torch.Size([3, 7])
    """
    if isinstance(data, Tensor):
        return data

    batches = default_input_mapping(data)
    # cannot use .flatten(1) because of tensors with shape (B, )
    batches = [batch.view(batch.size(0), -1).float() for batch in batches]
    combined = torch.cat(batches, 1)  # combined batch has shape (B, N)
    return combined


@under_review()
def collect_tensors(data: Any) -> List[Tensor]:
    """Filters all tensors in a collection and returns them in a list."""
    tensors = []

    def collect_batches(tensor: Tensor) -> Tensor:
        tensors.append(tensor)
        return tensor

    apply_to_collection(data, dtype=Tensor, function=collect_batches)
    return tensors


@under_review()
@contextmanager
def selective_eval(model: nn.Module, layer_types: Iterable[Type[nn.Module]]) -> None:
    """A context manager that sets all requested types of layers to eval mode. This method uses an ``isinstance``
    check, so all subclasses are also affected.

    Args:
        model: A model which has layers that need to be set to eval mode.
        layer_types: The list of class objects for which all layers of that type will be set to eval mode.
    """
    to_revert = []
    try:
        for module in model.modules():
            if isinstance(module, tuple(layer_types)):
                if module.training:
                    module.eval()
                    to_revert.append(module)
        yield
    finally:
        for module in to_revert:
            module.train()
