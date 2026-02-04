from __future__ import annotations
import torch


class BinarySTE(torch.autograd.Function):
    """Straight-through estimator (STE) to binarize continuous densities.

    """
    @staticmethod
    def forward(ctx, prob: torch.Tensor) -> torch.Tensor:
        """Threshold continuous densities into {0,1}.

        Parameters
        ----------
        prob : (1, 1, 32, 32) torch.Tensor
            Continuous densities (⍴ᵢⱼ∈[0,1]).

        Returns
        -------
        rho_binary : (1, 1, 32, 32) torch.Tensor
            Binary densities (⍴ᵢⱼ∈{0,1}).

        """
        return (prob > 0.5).float()

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        """Pass the gradient as is. 

        Treats the hard threshold in `forward()`
        as the identity for gradient flow.

        """
        return g
