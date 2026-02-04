from __future__ import annotations
import torch
from torch import nn


def smoothness_penalty(inputs: torch.Tensor) -> torch.Tensor:
    """Smoothness penalty function.

    Parameters
    ----------
    inputs : (1, 1, 32, 32) torch.Tensor
        Microstructure design.

    Returns
    -------
    penalty : (1,) torch.Tensor
        Smoothness penalty.

    """
    inputs_padded = nn.functional.pad(input=inputs, pad=(1, 1, 1, 1), mode='circular')

    # Moore neighborhood
    kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32, device=inputs.device)

    # Size: (1, 1, 3, 3)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    laplacian = nn.functional.conv2d(input=inputs_padded, weight=kernel)

    return torch.mean(input=laplacian ** 2)
