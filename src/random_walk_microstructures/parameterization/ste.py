from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class BinarySTE(torch.autograd.Function):
    """Straight-through estimator (STE) to binarize continuous densities.

    """
    @staticmethod
    def forward(ctx: FunctionCtx, prob: torch.Tensor) -> torch.Tensor:
        """Threshold continuous densities into {0,1}.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object to store information for backward pass.
        prob : (1, 1, 32, 32) torch.Tensor
            Continuous densities.

        Returns
        -------
        rho_binary : (1, 1, 32, 32) torch.Tensor
            Binary densities.

        """
        return (prob > 0.5).float()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass the gradient as is. 

        Treats the hard threshold in `forward()`
        as the identity for gradient flow.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object with stored information for backward pass.
        grad_output : (1, 1, 32, 32) torch.Tensor
            Gradient of the loss with respect to the binary densities.

        """
        return grad_output
