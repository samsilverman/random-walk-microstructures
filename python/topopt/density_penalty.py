from __future__ import annotations
import torch


def density_penalty(inputs: torch.Tensor, target_density) -> torch.Tensor:
    """Density penalty function.

    Parameters
    ----------
    inputs : (1, 1, 32, 32) torch.Tensor
        Microstructure design.

    Returns
    -------
    penalty : (1,) torch.Tensor
        Density penalty.

    """
    penalty = (torch.mean(input=inputs) - target_density) ** 2

    return penalty
