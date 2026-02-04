from __future__ import annotations
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
import torch
from torch import nn
from utils import InvertibleColumnTransformer


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Sign preserving `np.log1p`.

    """
    return np.sign(x) * np.log1p(np.abs(x))


def signed_log1p_torch(x: torch.Tensor) -> torch.Tensor:
    """A PyTorch analogue of `signed_log1p`.

    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    """Sign preserving `np.expm1`.

    """
    return np.sign(x) * np.expm1(np.abs(x))


def signed_expm1_torch(x: torch.Tensor) -> torch.Tensor:
    """A PyTorch analogue of `signed_expm1`.

    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def get_outputs_processor() -> InvertibleColumnTransformer:
    """Data preprocessor for homogenized stiffness tensor coefficients.

    Separate preprocessor pipelines are used for diagonal (C̄₁₁, C̄₂₂, C̄₃₃) and
    off-diagonal (C̄₁₂, C̄₁₃, C̄₂₃) coefficients.

    Returns
    -------
    processor : InvertibleColumnTransformer
        Data preprocessor for homogenized stiffness tensor coefficients.

    """
    processor = InvertibleColumnTransformer([
        ("diagonal", Pipeline([
            ('log1p', FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=False)),
            ('standard', StandardScaler())]), [0, 1, 2]),
        ("off-diagonal", Pipeline([
            ('signed-log1p', FunctionTransformer(func=signed_log1p, inverse_func=signed_expm1, check_inverse=False)),
            ('standard', StandardScaler())]), [3, 4, 5]),
    ])

    return processor


class OutputsProcessor(nn.Module):
    """A PyTorch analogue of the processor from `get_outputs_processor()`.

    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: torch.Tensor) -> OutputsProcessor:
        """Fit the `OutputsProcessor` using `x`.

        Parameters
        ----------
        x : (N, 6) torch.Tensor
            Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

        Returns
        -------
        self : OutputsProcessor
            Fitted processor instance.

        """
        diagonal = x[:, [0, 1, 2]]
        off_diagonal = x[:, [3, 4, 5]]

        diagonal = torch.log1p(diagonal)
        off_diagonal = signed_log1p(off_diagonal)

        diagonal_mean = diagonal.mean(dim=0, keepdim=True)
        diagonal_std = diagonal.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)

        off_diagonal_mean = off_diagonal.mean(dim=0, keepdim=True)
        off_diagonal_std = off_diagonal.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)

        # register as buffers so they move with .to(device)
        self.register_buffer('diagonal_mean_',  diagonal_mean)
        self.register_buffer('diagonal_std_', diagonal_std)
        self.register_buffer('off_diagonal_mean_',   off_diagonal_mean)
        self.register_buffer('off_diagonal_std_',  off_diagonal_std)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transfroms raw data to normalized data.

        Parameters
        ----------
        x : (N, 6) torch.Tensor
            Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

        Returns
        -------
        xᵗ : (N, 6) torch.Tensor
            Transformed homogenized stiffness tensor coefficients.

        """
        diagonal = x[:, [0, 1, 2]]
        off_diagonal = x[:, [3, 4, 5]]

        diagonal = torch.log1p(input=diagonal)
        off_diagonal = signed_log1p(x=off_diagonal)

        # Standardize
        diagonal = (diagonal - self.diagonal_mean_) / self.diagonal_std_
        off_diagonal = (off_diagonal - self.off_diagonal_mean_) / self.off_diagonal_std_

        return torch.cat([diagonal, off_diagonal], dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Transfroms normalized data to raw data.

        Parameters
        ----------
        xᵗ : (N, 6) torch.Tensor
            Transformed homogenized stiffness tensor coefficients.

        Returns
        -------
        x : (N, 6) torch.Tensor
            Homogenized stiffness tensor coefficients (C̄₁₁, C̄₂₂, C̄₃₃, C̄₁₂, C̄₁₃, C̄₂₃).

        """
        diagonal = x[:, [0, 1, 2]]
        off_diagonal = x[:, [3, 4, 5]]

        # Undo standardize
        diagonal = diagonal * self.diagonal_std_ + self.diagonal_mean_
        off_diagonal = off_diagonal * self.off_diagonal_std_ + self.off_diagonal_mean_

        diagonal = torch.expm1(input=diagonal)
        off_diagonal = signed_expm1_torch(x=off_diagonal)

        return torch.cat([diagonal, off_diagonal], dim=-1)
