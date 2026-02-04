#!/usr/bin/env python3
"""Script for microstructure design CSV → OBJ export.

Given a binary microstructure design stored as a CSV, this script
performs the following pipeline:

1. Loads design from CSV
2. Tiles the design
3. Removes disconnected components from the tiling
4. Upsamples and smooths the tiled design
5. Extracts contours via Marching Squares
6. Triangulates the contours to form an OBJ mesh

Parameters are currently set in `main()`.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
from skimage import measure, filters
import triangle as tr


def main() -> None:
    # Number of unit-cell tiles in the x-direction (horizontal repetition)
    nx = 6

    # Number of unit-cell tiles in the y-direction (vertical repetition)
    ny = 6

    # Extrusion thickness (OBJ z-height)
    # Set to ¼ that of the x- and y-dimension sizes
    t = (32 * nx) // 4

    save_dir = Path(__file__).resolve().parent.parent / 'data' / 'topopt_designs'

    # Input CSV file
    in_file = save_dir / 'ortho_80.csv'

    # Output OBJ file
    out_file = save_dir / 'ortho_80.obj'

    ###########################
    ########## Setup ##########
    ###########################
    cmap = LinearSegmentedColormap.from_list(name='white_to_C0', colors=['white', 'C0'], N=256)

    _, axes = plt.subplots(nrows=1, ncols=6, constrained_layout=True, figsize=(2 * 6.4, 0.5 * 4.8))

    #########################################
    ########## Step 1: Load design ##########
    #########################################
    design = np.loadtxt(fname=in_file, delimiter=',', dtype=float)

    axes[0].imshow(design, cmap=cmap, origin='upper')
    axes[0].axis('off')
    axes[0].set_title('Design')

    ##################################
    ########## Step 2: Tile ##########
    ##################################
    tiled = np.tile(A=design, reps=(nx, ny))

    axes[1].imshow(tiled, cmap=cmap, origin='upper')
    axes[1].axis('off')
    axes[1].set_title('Tiled')

    ############################################################
    ########## Step 3: Remove disconnected components ##########
    ############################################################

    labels = measure.label(label_image=tiled, background=0.0, connectivity=1)
    counts = np.bincount(labels.ravel())

    # Zero out the background count so it’s not chosen
    counts[0] = 0

    largest_label = counts.argmax()
    largest_region = labels == largest_label

    axes[2].imshow(largest_region, cmap=cmap, origin='upper')
    axes[2].axis('off')
    axes[2].set_title('Filtered')

    ###############################################
    ########## Step 4: Upsample & smooth ##########
    ###############################################

    # pad so border is correctly found
    padded = np.pad(array=largest_region, pad_width=1, mode='constant', constant_values=0)

    upsampled = np.repeat(a=np.repeat(a=padded, repeats=1, axis=0), repeats=1, axis=1)
    smooth = filters.gaussian(image=upsampled, sigma=1)

    axes[3].imshow(smooth, cmap=cmap, origin="upper")
    axes[3].axis('off')
    axes[3].set_title('Upsampled & Smoothed')

    ##############################################
    ########## Step 5: Marching squares ##########
    ##############################################

    contours = measure.find_contours(image=smooth, level=0.5)

    # Sort so largest (i.e. entire border) is first
    contours = sorted(contours, key=lambda contour: contour.shape[0], reverse=True)

    # swap so (x,y) instead of (y,x)
    contours = [contour[:, [1, 0]] for contour in contours]

    # reduce size
    contours = [measure.approximate_polygon(coords=contour, tolerance=0.1) for contour in contours]

    axes[4].imshow(upsampled, cmap=cmap, origin="upper")
    for contour in contours:
        axes[4].plot(contour[:, 0], contour[:, 1], color='C1')
    axes[4].axis('off')
    axes[4].set_title('Contours')

    #########################################
    ########## Step 6: Triangulate ##########
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

    axes[5].add_collection(
        PolyCollection(
            verts=B['vertices'][B['triangles']],
            edgecolors='black',
            facecolors='C0'
        )
    )
    axes[5].set_xlim(axes[4].get_xlim())
    axes[5].set_ylim(axes[4].get_ylim())
    axes[5].set_aspect('equal')
    axes[5].axis('off')
    axes[5].set_title('Triangulation')

    plt.show()

    #########################################
    ########## Step 7: Save as OBJ ##########
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

    plt.show()


if __name__ == "__main__":
    main()
