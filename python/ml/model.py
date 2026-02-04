from __future__ import annotations
from typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor, device


class CNN(nn.Module):
    """Convolutional neural network (CNN) model.

    """

    def __init__(self) -> None:
        """Initialize CNN.

        """
        super().__init__()

        self.layers_ = nn.Sequential(
            nn.CircularPad2d(padding=6),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=6)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        inputs : (N, 1, 32, 32) torch.Tensor
            Microstructure designs.

        Returns
        -------
        outputs : (N, 6) torch.Tensor
            Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

        """
        outputs = self.layers_(inputs)

        return outputs

    def device(self) -> device:
        """The device the model is currently on.

        """
        return next(self.parameters()).device
