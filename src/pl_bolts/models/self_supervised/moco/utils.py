from typing import Any, List, Tuple

import torch
from torch import Tensor


def validate_batch(batch: Tuple[List[List[Tensor]], List[Any]]) -> Tensor:
    """Reads a batch of data, validates the format, and stacks the images into a single tensor.

    Contrastive SSL models expect each image in the batch to be transformed in multiple (typically two) ways. The input
    is similar to image classification and object detection models, but a tuple of images is expected in place of each
    image.

    Args:
        batch: The batch of data read by the :class:`~torch.utils.data.DataLoader`. A tuple containing a nested list of
            ``N`` image pairs (or tuples of more than two images) and a list of ``N`` target dictionaries.

    Returns:
        The input batch with images stacked into a single ``[N, 2, channels, height, width]`` tensor.

    """
    images, targets = batch

    if not images:
        raise ValueError("No images in batch.")

    batch_size = len(images)
    if batch_size != len(targets):
        raise ValueError(f"Got {batch_size} image pairs, but {len(targets)} targets.")

    image_transforms = images[0]

    if isinstance(image_transforms, Tensor):
        if image_transforms.ndim != 4:
            raise ValueError(
                "Contrastive training expects the transformed images as a tuple, a list, or a 4-dimensional tensor. "
                f"Got a tensor with {image_transforms.ndim} dimensions."
            )
        shape = image_transforms.shape
        for image_transforms in images[1:]:
            if not isinstance(image_transforms, Tensor):
                raise ValueError(f"Expected transformed images in a tensor, got {type(image_transforms)}.")
            if image_transforms.shape != shape:
                raise ValueError(
                    f"Different shapes for transformed images in one batch: {shape} and {image_transforms.shape}"
                )

        return torch.stack(images)

    if isinstance(image_transforms, (tuple, list)):
        num_transforms = len(image_transforms)
        if num_transforms < 2:
            raise ValueError(
                f"Contrastive training expects at least two transformations of every image, got {num_transforms}."
            )
        if not isinstance(image_transforms[0], Tensor):
            raise ValueError(f"Expected image to be of type Tensor, got {type(image_transforms[0]).__name__}.")
        shape = image_transforms[0].shape
        for image_transforms in images:
            for image in image_transforms:
                if not isinstance(image, Tensor):
                    raise ValueError(f"Expected image to be of type Tensor, got {type(image).__name__}.")
                if image.shape != shape:
                    raise ValueError(f"Images with different shapes in one batch: {shape} and {image.shape}")

        # PyTorch doesn't stack nested lists of tensors. Stacking the tensors in two steps would cause the data to be
        # copied twice, so instead we'll first flatten the hierarchy and then reshape in the end.
        flat_images = [image for image_transforms in images for image in image_transforms]
        flat_images = torch.stack(flat_images)  # [batch_size * num_transforms, channels, height, width]
        return flat_images.view(batch_size, num_transforms, *shape)

    raise ValueError(
        "Contrastive training expects the transformed images as a tuple, a list, or a 4-dimensional tensor. Got "
        f"{type(image_transforms).__name__}."
    )


class ConcatenateAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor) -> Tensor:  # type: ignore
        """Concatenates tensors from all GPUs."""
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor.contiguous())
        return torch.cat(gathered_tensor, 0)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # type: ignore
        """Sums the gradients from all GPUs and takes the ones corresponding to our mini-batch."""
        start_idx = torch.distributed.get_rank() * ctx.batch_size
        stop_idx = start_idx + ctx.batch_size

        grad_input = grad_output.clone().contiguous()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM)
        return grad_input[start_idx:stop_idx]


@torch.no_grad()
def concatenate_all(tensor: Tensor) -> Tensor:
    """Performs ``all_gather`` operation to concatenate the provided tensor from all devices.

    This function has no gradient.
    """
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_tensor, tensor.contiguous())
    return torch.cat(gathered_tensor, 0)


@torch.no_grad()
def shuffle_batch(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Redistributes the batch randomly to different devices.

    Gathers a mini-batch from all devices and shuffles it into a random order. Each device will receive a random subset
    of the mini-batch. Only support Distributed Data Parallel (DDP) training strategy.

    Args:
        x: The input tensor, whose first dimension is the batch.

    Returns:
        The output tensor and a list of indices that gives the original order of the combined mini-batch. The output
        tensor is the same size as the input tensor, but contains a random subset of the combined mini-batch.

    """
    all_x = concatenate_all(x)

    local_batch_size = x.shape[0]
    global_batch_size = all_x.shape[0]
    num_gpus = global_batch_size // local_batch_size

    # Create a random ordering of the images in all GPUs and broadcast it from rank 0 to the other GPUs.
    random_order = torch.randperm(global_batch_size).cuda()
    torch.distributed.broadcast(random_order, src=0)

    # Save a mapping from the shuffled order back to the linear order.
    original_order = torch.argsort(random_order)

    rank = torch.distributed.get_rank()
    local_idxs = random_order.view(num_gpus, -1)[rank]
    return all_x[local_idxs], original_order


@torch.no_grad()
def sort_batch(x: Tensor, order: Tensor) -> Tensor:
    """Sorts the samples across devices into given order.

    Gathers a mini-batch from all devices and sorts it into given order. Each device will receive a consecutive subset
    of the mini-batch. Only support Distributed Data Parallel (DDP) training strategy.

    Args:
        x: The input tensor, whose first dimension is the batch.
        order: Indices to the combined mini-batch in the correct order.

    Returns:
        The subset of the combined mini-batch that corresponds to this device.

    """
    all_x = concatenate_all(x)

    local_batch_size = x.shape[0]
    global_batch_size = all_x.shape[0]
    num_gpus = global_batch_size // local_batch_size

    rank = torch.distributed.get_rank()
    local_idxs = order.view(num_gpus, -1)[rank]
    return all_x[local_idxs]
