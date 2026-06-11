#!/usr/bin/env python
from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
import numpy as np
from skimage import measure, filters
import triangle as tr

CURR_DIR = Path(__file__).resolve().parent


def build_parser() -> ArgumentParser:
    """Command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Command-line interface.

    """
    parser = ArgumentParser(description='Export microstructure design stored in CSV file to an OBJ mesh.', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n', type=int, default=1, help='Number of repetitions to tile the design in each direction.')
    parser.add_argument('--t', type=float, default=0.25, help='Extrusion thickness as a fraction of side length.')
    parser.add_argument('--file', type=Path, default=Path('results').resolve() / 'design.csv', help='CSV file.')
    parser.add_argument('--visualize', action='store_true', help='Visualize results.')

    return parser


def main() -> None:
    """Export microstructure design stored in CSV file to an OBJ mesh from the command line.

    """
    #############################################
    ########## CLI argument validation ##########
    #############################################

    parser = build_parser()
    args = parser.parse_args()

    if args.n <= 0:
        parser.error(f'--n must be positive, got {args.n}.')

    if args.t <= 0:
        parser.error(f'--t must be positive, got {args.t}.')

    ###########################
    ########## Setup ##########
    ###########################

    # Extrusion thickness (OBJ z-height)
    # Set to ¼ that of the x- and y-dimension sizes
    t = (32 * args.n) // 4

    # Input CSV file
    in_file = Path(args.file).resolve()

    # Output OBJ file
    out_file = in_file.with_suffix('.obj')

    cmap = LinearSegmentedColormap.from_list(name='white_to_C0', colors=['white', 'C0'], N=256)

    design = np.loadtxt(fname=in_file, delimiter=',', dtype=float)

    ##################################
    ########## Step 1: Tile ##########
    ##################################

    # Step 1: Tile
    tiled = np.tile(A=design, reps=(args.n, args.n))

    ############################################################
    ########## Step 2: Remove disconnected components ##########
    ############################################################

    labels = measure.label(label_image=tiled, background=0.0, connectivity=1)
    counts = np.bincount(labels.ravel())

    # Zero out the background count so it’s not chosen
    counts[0] = 0

    largest_label = counts.argmax()
    largest_region = labels == largest_label

    ###############################################
    ########## Step 3: Upsample & smooth ##########
    ###############################################

    # pad so border is correctly found
    padded = np.pad(array=largest_region, pad_width=1, mode='constant', constant_values=0)

    upsampled = np.repeat(a=np.repeat(a=padded, repeats=1, axis=0), repeats=1, axis=1)
    smooth = filters.gaussian(image=upsampled, sigma=1)

    ##############################################
    ########## Step 4: Marching squares ##########
    ##############################################

    contours = measure.find_contours(image=smooth, level=0.5)

    # Sort so largest (i.e. entire border) is first
    contours = sorted(contours, key=lambda contour: contour.shape[0], reverse=True)

    # swap so (x,y) instead of (y,x)
    contours = [contour[:, [1, 0]] for contour in contours]

    # reduce size
    contours = [measure.approximate_polygon(coords=contour, tolerance=0.1) for contour in contours]

    #########################################
    ########## Step 5: Triangulate ##########
    #########################################

    vertices = []
    segments = []
    holes = []
    offset = 0

    for i, contour in enumerate(contours):
        # remove last point if last = first
        if np.array_equal(a1=contour[0], a2=contour[-1]):
            contour = contour[:-1]

        vertices.append(contour)

        contour_segements = np.column_stack([
            # x
            np.arange(start=0, stop=contour.shape[0]),
            # y
            np.roll(a=np.arange(start=0, stop=contour.shape[0]), shift=-1)
        ])

        segments.append(contour_segements + offset)
        offset += contour.shape[0]

        if i > 0:
            holes.append(np.mean(contour, axis=0).tolist())

    vertices = np.vstack(vertices)
    segments = np.vstack(segments)

    A = dict(vertices=vertices, segments=segments)
    if len(holes) > 0:
        A = dict(vertices=vertices, segments=segments, holes=holes)
    B = tr.triangulate(A, 'pq')

    #########################################
    ########## Step 6: Save as OBJ ##########
    #########################################

    with open(file=out_file, mode='w', encoding='utf-8') as file:
        # bottom then top vertices
        for vertex in B['vertices']:
            x = vertex[0]
            y = vertex[1]
            z = 0
            file.write(f'v {x} {y} {z}\n')
            z = t
            file.write(f'v {x} {y} {z}\n')

        # bottom then top triangles
        for triangle in B['triangles']:
            i = 2 * triangle[0] + 1
            j = 2 * triangle[1] + 1
            k = 2 * triangle[2] + 1
            file.write(f'f {k} {j} {i}\n')

            i += 1
            j += 1
            k += 1
            file.write(f'f {i} {j} {k}\n')

        # side triangles
        for segment in segments:
            bottom_i = 2 * segment[0] + 1
            bottom_j = 2 * segment[1] + 1
            top_i = bottom_i + 1
            top_j = bottom_j + 1

            file.write(f'f {bottom_i} {bottom_j} {top_j}\n')
            file.write(f'f {bottom_i} {top_j} {top_i}\n')

    ##############################
    ########## Plotting ##########
    ##############################

    if not args.visualize:
        return

    _, axes = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(6.4, 4.8))

    axes[0, 0].imshow(design, cmap=cmap, origin='upper')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Design')

    axes[0, 1].imshow(tiled, cmap=cmap, origin='upper')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Tiled')

    axes[0, 2].imshow(largest_region, cmap=cmap, origin='upper')
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Filtered')

    axes[1, 0].imshow(smooth, cmap=cmap, origin="upper")
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Upsampled & Smoothed')

    axes[1, 1].imshow(upsampled, cmap=cmap, origin="upper")
    for contour in contours:
        axes[1, 1].plot(contour[:, 0], contour[:, 1], color='C1')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Contours')

    axes[1, 2].add_collection(
        PolyCollection(
            verts=B['vertices'][B['triangles']],
            edgecolors='black',
            facecolors='C0'
        )
    )
    axes[1, 2].set_xlim(axes[1, 1].get_xlim())
    axes[1, 2].set_ylim(axes[1, 1].get_ylim())
    axes[1, 2].set_aspect('equal')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Triangulation')

    plt.show()


if __name__ == "__main__":
    main()
