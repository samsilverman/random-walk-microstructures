#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import get_args, Literal, TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from random_walk_microstructures import DEFAULT_DTYPE
from random_walk_microstructures import set_seed, load_data, split_indices, OutputsProcessor, CNN, Design
import torch
from torch import nn
from torch import optim
from tqdm import trange

if TYPE_CHECKING:
    from argparse import Namespace

DTYPE = DEFAULT_DTYPE
DEVICE = 'cpu'

Objective = Literal['iso', 'ortho']


def lerp(l0: float, l1: float, step: int, num_steps: int) -> float:
    """Linearly interpolate between two values over a fixed number of steps.

    Parameters
    ----------
    l0 : float
        Start value.
    l1 : float
        End value.
    step : int
        Current step.
    num_steps : int
        Total number of steps in the schedule.

    Returns
    -------
    value : float
        Interpolated value.

    """
    if num_steps <= 1:
        return float(l1)

    t = step / (num_steps - 1)

    # Clamp t to [0,1] so we don't extrapolate outside [l0,l1].
    t = max(0.0, min(1.0, float(t)))

    return l0 + (l1 - l0) * t


def objective_score(C: torch.Tensor, objective: Objective) -> torch.Tensor:
    """Topology optimization objective score.

    Parameters
    ----------
    C : (3, 3) torch.Tensor
        Stiffness tensor.
    objective : Objective
        Objective type.

    Returns
    -------
    score : (1,) torch.Tensor
        Score.

    """
    S = torch.inverse(input=C)

    if objective == 'iso':
        return S[0, 0] + S[1, 1]

    if objective == 'ortho':
        return 0.5 * (S[1, 1] / S[0, 0] - 3) ** 2
        # return 0.5 * (S[1,1] / S[0,0] - 3.0001) ** 2

    raise ValueError(f'Unsupported objective ({objective}).')


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


def tile_design(design: torch.Tensor) -> torch.Tensor:
    """Tile a microstructure design into a 3x3 mosaic with dimmed neighbors.

    Parameters
    ----------
    design : (32, 32) torch.Tensor
        Microstructure design.

    Returns
    -------
    tiled : (96, 96) torch.Tensor
        Tiled microstructure design (with dimmed neighbors).

    """
    light = 0.5 * design

    row0 = torch.cat([light, light, light], dim=1)
    row1 = torch.cat([light, design, light], dim=1)
    row2 = torch.cat([light, light, light], dim=1)

    return torch.cat([row0, row1, row2], dim=0)


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Topology optimization.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--vf', type=float, required=True, help='Target volume fraction.')
    parser.add_argument('--objective', choices=get_args(Objective), default='iso', help='Topology optimization objective function.')
    parser.add_argument('--dir', type=Path, default=Path('results'), help='Directory containing the saved model.')
    parser.add_argument('--out-dir', type=Path, default=Path('results'), help='Output directory for the saved results.')
    parser.add_argument('--smooth-start', type=float, default=1.0, help='Initial weight of the smoothness penalization.')
    parser.add_argument('--smooth-end', type=float, default=100.0, help='Final weight of the smoothness penalization.')
    parser.add_argument('--density-start', type=float, default=1.0, help='Initial weight of the density penalization.')
    parser.add_argument('--density-end', type=float, default=100.0, help='Final weight of the density penalization.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=3019, help='RNG seed.')
    parser.add_argument('--visualize', action='store_true', help='Visualize optimization.')

    return parser


def validate_cli_arguments(args: Namespace, parser: ArgumentParser) -> None:
    """Validate command-line interface (CLI) arguments.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.
    parser : argparse.ArgumentParser
        Parser used to raise errors.

    """
    if args.vf < 0 or args.vf > 1:
        parser.error(f'--vf must be in [0, 1], got {args.vf}.')

    if args.epochs <= 0:
        parser.error(f'--epochs must be positive, got {args.epochs}.')

    if args.smooth_start <= 0:
        parser.error(f'--smooth_start must be positive, got {args.smooth_start}.')

    if args.smooth_end < args.smooth_start:
        parser.error(f'--smooth_end must be larger than {args.smooth_start}, got {args.smooth_end}.')

    if args.density_start <= 0:
        parser.error(f'--density_start must be positive, got {args.density_start}.')

    if args.density_end < args.density_start:
        parser.error(f'--smooth_end must be larger than {args.density_start}, got {args.density_end}.')

    if args.epochs <= 0:
        parser.error(f'--epochs must be positive, got {args.epochs}.')

    if args.lr <= 0:
        parser.error(f'--lr must be positive, got {args.lr}.')

    max_seed = 2**32 - 1
    if args.seed < 0 or args.seed > max_seed:
        parser.error(f'--seed must be in [0, {max_seed}], got {args.seed}.')


def main() -> None:
    """Topology optimization from the command line.

    """
    #############################################
    ########## CLI argument validation ##########
    #############################################

    parser = build_parser()
    args = parser.parse_args()

    validate_cli_arguments(args=args, parser=parser)

    ###########################
    ########## Setup ##########
    ###########################

    set_seed(seed=args.seed)

    # Data preprocessing
    _, outputs = load_data()

    train_indices, _, _ = split_indices(num_samples=outputs.shape[0], percent_train=0.9, percent_valid=0.05)

    outputs_train = outputs[train_indices]

    outputs_train = torch.tensor(outputs_train, dtype=DTYPE, device=DEVICE)

    outputs_processor = OutputsProcessor().to(dtype=DTYPE, device=DEVICE)
    outputs_processor.fit(x=outputs_train)

    # Model
    model = CNN().to(dtype=DTYPE, device=DEVICE)

    state_dict = torch.load(f=args.dir / 'model.pt', map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Design
    design = Design().to(dtype=DTYPE, device=DEVICE)

    # Optimizer
    optimizer = optim.Adam(design.parameters(), lr=args.lr)

    # Live visualization setup
    cmap = LinearSegmentedColormap.from_list(name='white_to_C0', colors=['white', 'C0'], N=256)

    if args.visualize:
        plt.ion()

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

        # Get inital density field
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

    ###########################################
    ########## Topology optimization ##########
    ###########################################

    best_loss = float('inf')
    best_design = None
    best_objective = None
    best_smooth = None
    best_density = None

    progress_bar = trange(args.epochs)

    for epoch in progress_bar:
        rho_binary, rho_cont = design()
    
        out = model(rho_binary)
        out = outputs_processor.inverse(x=out)
    
        C11, C22, C33, C12, C13, C23 = out[0]
    
        C = torch.stack([torch.stack([C11, C12, C13]),
                         torch.stack([C12, C22, C23]),
                         torch.stack([C13, C23, C33])])

        lambda_smooth = lerp(l0=args.smooth_start, l1=args.smooth_end, step=epoch, num_steps=args.epochs)
        lambda_density = lerp(l0=args.density_start, l1=args.density_end, step=epoch, num_steps=args.epochs)

        objective = objective_score(C=C, objective=args.objective)
        smooth = lambda_smooth * smoothness_penalty(inputs=rho_cont)
        density = lambda_density * density_penalty(inputs=rho_binary, target_density=args.vf)

        loss = objective + smooth + density

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_design = rho_binary.detach().cpu().squeeze()
            best_objective = objective.item()
            best_smooth = smooth.item()
            best_density = density.item()

            if args.visualize:
                tiled = tile_design(design=best_design)
                tiled_image.set_data(A=tiled)

                fig.canvas.draw()
                plt.pause(interval=0.001)

        progress_bar.set_postfix(objective=f'{best_objective:.2e}', smooth=f'{best_smooth:.2e}', density=f'{best_density:.2e}')

    best_design = best_design.cpu().numpy()
    best_design = np.flipud(m=best_design)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(fname=args.out_dir / 'design.csv', X=best_design, delimiter=',', fmt='%d')

    if args.visualize:
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
