from typing import Callable

import torch
from torch import Tensor


def conjugate_gradient(A: Callable, b: Tensor, delta: float = 0.0, max_iterations: int = 10) -> Tensor:
    """
    Conjugate Gradient iteration to solve Ax = b.
    It solves Ax=b without forming the full matrix, just compute the matrix-vector product (The Fisher-vector product)

    Args:
        A: callable which calculates fisher product
        b: Tensor, right-hand side of the linear system.
        delta: float value indicating the tolerance to a change in value. If the change in the target value is less
            than this value, the calculation is stopped.
        max_iterations: int value indicating maximum number of iterations. Iteration will stop after maxiter steps
            even if the specified tolerance has not been achieved.

    Returns:
        Tensor value with the converged solution.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    for _ in range(max_iterations):
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        r = r - alpha * AVP
        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x
