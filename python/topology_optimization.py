#!/usr/bin/env python3
"""Microstructure topology optimization.

"""
from __future__ import annotations
from typing import Tuple, Callable, List
from pathlib import Path
import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from utils import set_seed, load_state_dict, split_indices
from ml import load_data, CNN, OutputsProcessor
from topopt import density_penalty, smoothness_penalty, lerp, TopOptDesign, tile_design


def topology_optimization(epochs: int,
                          lambda_smooth: Tuple[float, float],
                          lambda_density: Tuple[float, float],
                          target_density: float,
                          objective_function: Callable[[torch.Tensor], torch.Tensor],
                          verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Gradient-based topology optimization using a trained CNN surrogate.

    Parameters
    ----------
    epochs : int
        Number of topology optimization epochs.
    lambda_smooth : Tuple[float, float]
        (start, end) weight for the smoothness penalty. The current value is
        linearly interpolated across epochs.
    lambda_density : Tuple[float, float]
        (start, end) weight for the density penalty. The current value is
        linearly interpolated across epochs.
    target_density : float
        Target mean density (volume fraction) of the optimized design.
    objective_function : Callable[[torch.Tensor], torch.Tensor]
        Function mapping a 3x3 stiffness tensor C to a scalar objective to minimize.
    verbose : bool (default=`False`)
        Set to `True` to see status messages during topology optimization.

    Returns
    -------
    best_design : (32, 32) torch.Tensor
        Optimal microstructure design.
    best_out
        Optimal microstructure design's predicted homogenized stiffness tensor coefficients.
    objectives : List[float]
        Topology optimization objective values at each epoch.

    """
    # Setup
    seed = 3019
    set_seed(seed)

    # Data preprocessing
    _, outputs = load_data()

    train_indices, _, _ = split_indices(num_samples=outputs.shape[0], percent_train=0.9, percent_valid=0.05)

    outputs_train = outputs[train_indices]

    outputs_train = torch.tensor(outputs_train, dtype=torch.float32)

    outputs_processor = OutputsProcessor()
    outputs_processor.fit(x=outputs_train)

    # Model
    model = CNN()

    model_dir = Path(__file__).resolve().parent.parent / 'models'
    load_state_dict(model=model, file=model_dir / 'model.pt')

    design = TopOptDesign()

    model.to(device='cpu')
    outputs_processor.to(dtype=torch.float32, device=model.device())
    design.to(dtype=torch.float32, device=model.device())

    # Live figures
    cmap = LinearSegmentedColormap.from_list(name='white_to_C0', colors=['white', 'C0'], N=256)
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

    rho_binary, _ = design()
    rho_binary = rho_binary.detach().cpu().squeeze()
    tiled = tile_design(design=rho_binary)

    tiled_image = ax.imshow(X=tiled, vmin=0, vmax=1, cmap=cmap, origin='lower')

    rect = Rectangle(xy=(31.5, 31.5), width=32, height=32, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(p=rect)

    zoom = 16
    ax.set_xlim([zoom, tiled.shape[0]-zoom])
    ax.set_ylim([zoom, tiled.shape[0]-zoom])
    ax.axis('off')

    # Optimization loop
    best_loss = float('inf')
    best_design = None
    best_out = None
    best_objective = None
    best_smooth = None
    best_density = None
    objectives = []

    optimizer = optim.Adam(design.parameters(), lr=0.01)

    if verbose:
        print(f'{"-" * 5}Topology Optimization Start{"-" * 5}')

    for epoch in range(epochs):
        rho_binary, rho_cont = design()

        out = model(rho_binary)
        out = outputs_processor.inverse(x=out)

        C11, C22, C33, C12, C13, C23 = out[0]

        C = torch.stack([torch.stack([C11, C12, C13]),
                         torch.stack([C12, C22, C23]),
                         torch.stack([C13, C23, C33])])

        curr_lambda_smooth = lerp(l0=lambda_smooth[0], l1=lambda_smooth[1], step=epoch, num_steps=epochs)
        curr_lambda_density = lerp(l0=lambda_density[0], l1=lambda_density[1], step=epoch, num_steps=epochs)

        objective = objective_function(C)
        smooth  = curr_lambda_smooth * smoothness_penalty(inputs=rho_cont)
        density = curr_lambda_density * density_penalty(inputs=rho_binary, target_density=target_density)

        loss = objective + smooth + density
        objectives.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_design = rho_binary.detach().cpu().squeeze()
            best_out = out.detach().cpu().squeeze()

            best_objective = objective.item()
            best_smooth = smooth.item()
            best_density = density.item()

            tiled = tile_design(design=best_design)
            tiled_image.set_data(A=tiled)

            fig.canvas.draw()
            plt.pause(interval=0.001)

        if verbose and ((epoch + 1) % 100 == 0):
            print(f'epoch: {epoch + 1}/{epochs}, objective: {best_objective:.5f}, smooth: {best_smooth:.5f}, density: {best_density:.5f}')

    if verbose:
        print(f'{"-" * 5}Topology Optimization End{"-" * 5}')

    plt.ioff()
    plt.show()

    return best_design, best_out, objectives


def iso_objective(C: torch.Tensor) -> torch.Tensor:
    """Isotropic objective function.

    Parameters
    ----------
    C : (3, 3) torch.Tensor
        Homogenized stiffness tensor.

    Returns
    -------
    objective : (1,) torch.Tensor
        S₁₁+S₂₂ where S=C⁻¹.

    """
    S = torch.inverse(input=C)

    return S[0,0] + S[1,1]


def ortho_objective(C: torch.Tensor) -> torch.Tensor:
    """Orthotropic objective function.

    Parameters
    ----------
    C : (3, 3) torch.Tensor
        Homogenized stiffness tensor.

    Returns
    -------
    objective : (1,) torch.Tensor
        ½(S₁₁/S₂₂-3)² where S=C⁻¹.

    """
    S = torch.inverse(input=C)

    # return 0.5 * (S[1,1] / S[0,0] - 3.0001) ** 2
    return 0.5 * (S[1,1] / S[0,0] - 3) ** 2


def main():
    save_dir = Path(__file__).resolve().parent.parent / 'data' / 'topopt_designs'

    # Isotropic (density=0.8)
    inputs, _, _ = topology_optimization(epochs=10000,
                                         lambda_smooth=(1, 1000),
                                         lambda_density=(0.01, 1000),
                                         target_density=0.8,
                                         objective_function=iso_objective,
                                         verbose=True)

    np.savetxt(fname=save_dir / 'iso_80.csv', X=torch.flipud(input=inputs).numpy(), delimiter=',', fmt='%d')

    # Orthotropic (density=0.8)
    inputs, _, _ = topology_optimization(epochs=1000,
                                         lambda_smooth=(100, 1000),
                                         lambda_density=(0.01, 1000),
                                         target_density=0.8,
                                         objective_function=ortho_objective,
                                         verbose=True)

    np.savetxt(fname=save_dir / 'ortho_80.csv', X=torch.flipud(input=inputs).numpy(), delimiter=',', fmt='%d')


if __name__ == '__main__':
    main()
