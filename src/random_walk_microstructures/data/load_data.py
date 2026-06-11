from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the microstructure designs and homogenized stiffness tensor coefficients.

    Returns
    -------
    inputs : (131072, 1, 32, 32) numpy.ndarray
        Microstructure designs.
    outputs : (131072, 6) numpy.ndarray
        Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

    """
    curr_dir = Path(__file__).resolve().parent
    data_dir = curr_dir.parent.parent.parent / 'data'

    inputs = np.load(file=data_dir / 'inputs.npz')['data']
    outputs = np.load(file=data_dir / 'outputs.npz')['data']

    return inputs, outputs
