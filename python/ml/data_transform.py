from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import random
import torch


if TYPE_CHECKING:
    from torch import Tensor


def data_transform(inputs: Tensor, outputs: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply a random transformation to the data.

    Microstructure designs are randomly augmented using the following transformations:

    - Rotation: 0°, 90°, 180°, or 270°
    - Reflection: Horizontal flip
    - Translation: Cyclic shift in x and y

    Homogenized stiffness tensor coefficients are transformed accordingly in
    equivarient and invariant fashions.

    Parameters
    ----------
    inputs : (N, 1, 32, 32) torch.Tensor
        Microstructure designs.
    outputs : (N, 6) torch.Tensor
        Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

    Returns
    -------
    inputs_transform : (N, 1, 32, 32) torch.Tensor
        Transformed microstructure designs.
    outputs : (N, 6) torch.Tensor
        Transformed homogenized stiffness tensor coefficients.

    """
    inputs_transform = torch.empty_like(input=inputs)
    outputs_transform = torch.empty_like(input=outputs)

    # Random rotation
    k = random.choice([0, 1, 2, 3])

    inputs_transform = inputs.rot90(k=k, dims=(2, 3))

    # Random reflection
    reflect = random.choice([0, 1])

    if reflect:
        inputs_transform = inputs_transform.flip(dims=(3,))

    # Transform outputs accordingly
    outputs_transform = outputs

    if k in (1, 3) and reflect == 0:
        outputs_transform[:, [0, 1]] = outputs_transform[:, [1, 0]]
        outputs_transform[:, [4, 5]] = -outputs_transform[:, [5, 4]]
    elif k in (0, 2) and reflect == 1:
        outputs_transform[:, [4, 5]] *= -1
    elif k in (1, 3) and reflect == 1:
        outputs_transform[:, [0, 1]] = outputs_transform[:, [1, 0]]
        outputs_transform[:, [4, 5]] = outputs_transform[:, [5, 4]]

    x_trans = random.randint(0, inputs.shape[2] - 1)
    y_trans = random.randint(0, inputs.shape[3] - 1)
    inputs_transform = torch.roll(input=inputs_transform, shifts=(x_trans, y_trans), dims=(2, 3))

    return inputs_transform, outputs_transform
