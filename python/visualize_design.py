#!/usr/bin/env python3
"""Visualize a microstructure design stored in a CSV file.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main() -> None:
    save_dir = Path(__file__).resolve().parent.parent / 'data' / 'topopt_designs'

    # Input CSV file
    file = save_dir / 'ortho_80.csv'

    # Number of unit-cell tiles in the x-direction (horizontal repetition)
    nx = 6

    # Number of unit-cell tiles in the y-direction (vertical repetition)
    ny = 6

    design = np.loadtxt(fname=file, delimiter=",", dtype=int)

    cmap = ListedColormap(['white', 'C0'])

    _, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6.4, 4.8))

    tiled = np.tile(A=design, reps=(ny, nx))

    ax.imshow(X=tiled, cmap=cmap)
    ax.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
