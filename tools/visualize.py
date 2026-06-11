#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

CURR_DIR = Path(__file__).resolve().parent


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Visualize a microstructure design stored in a CSV file.', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n', type=int, default=1, help='Number of repetitions to tile the design in each direction.')
    parser.add_argument('--file', type=Path, default=Path('results').resolve() / 'design.csv', help='CSV file.')

    return parser


def main() -> None:
    """Visualize a microstructure design stored in a CSV file from the command line.

    """
    #############################################
    ########## CLI argument validation ##########
    #############################################

    parser = build_parser()
    args = parser.parse_args()

    if args.n <= 0:
        parser.error(f'--n must be positive, got {args.n}.')

    ###########################
    ########## Setup ##########
    ###########################

    design = np.loadtxt(fname=args.file, delimiter=',', dtype=int)

    cmap = ListedColormap(['white', 'C0'])

    ##############################
    ########## Plotting ##########
    ##############################

    _, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

    tiled = np.tile(A=design, reps=(args.n, args.n))

    ax.imshow(X=tiled, cmap=cmap, origin='upper')
    ax.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
