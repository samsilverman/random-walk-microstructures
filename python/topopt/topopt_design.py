from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from topopt import BinarySTE


class TopOptDesign(nn.Module):
    """Trainable microstructure densities for binary topology optimization.

    """
    def __init__(self) -> None:
        """Initialize TopOptDesign.

        """
        super().__init__()

        # 0 when passed through sigmoid become 0.5
        # So initialize densities to middle of [0,1] range
        init_rho_cont = torch.zeros(1, 1, 32, 32)

        self.rho_cont_ = nn.Parameter(init_rho_cont)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate continuous and binarized densities.

        Returns
        -------
        rho_binary : (1, 1, 32, 32) torch.Tensor
            Binary densities.
        rho_cont : (1, 1, 32, 32) torch.Tensor
            Continuous densities.

        """
        # Keeps continuous densities in [0,1] range
        rho_cont = torch.sigmoid(self.rho_cont_)
        rho_cont_rounded  = BinarySTE.apply(rho_cont)
        rho_binary = rho_cont + (rho_cont_rounded - rho_cont).detach()

        return rho_binary, rho_cont
