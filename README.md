# Random-Walk Microstructures for Diﬀerentiable Topology Optimization

[![Read the Paper](https://img.shields.io/static/v1.svg?label=DOI&message=10.1145/3745778.3766645&color=blue)](https://doi.org/10.1145/3745778.3766645)

![Teaser](https://github.com/samsilverman/random-walk-microstructures/blob/main/assets/images/teaser.png)

This repository contains the code for our ACM SCF 2025 paper “Random-Walk Microstructures for Differentiable Topology Optimization.”

   > Samuel Silverman, Dylan Balter, Keith A. Brown, Emily Whiting  
   > [*Random-Walk Microstructures for Differentiable Topology Optimization*](https://sam-silverman.com/assets/pdf/Silverman-RandomWalkMicrostructures.pdf)  
   > Proceedings of the  ACM Symposium on Computational Fabrication (2025)

## Data

The C++ libraries [rwmicro](https://github.com/samsilverman/rwmicro) and [monad](https://github.com/samsilverman/monad) provide the random-walk generation and homogenized stiffness tensor simulation used to construct the dataset.

`data/` contains:

- `inputs.npz`: (100000, 1, 32, 32) NumPy array of microstructure designs
- `outputs.npz`: (100000, 6) NumPy array of unique coefficients of the homogenized stiffness tensor ($`\overline{\mathbf{C}}_{11}`$, $`\overline{\mathbf{C}}_{22}`$, $`\overline{\mathbf{C}}_{33}`$, $`\overline{\mathbf{C}}_{12}`$, $`\overline{\mathbf{C}}_{13}`$, $`\overline{\mathbf{C}}_{23}`$).
These values were simulated assuming a base material with Young’s modulus $E=2100$ MPa and Poisson’s ratio $\nu=0.39$, corresponding to Formlabs Tough 2000 resin.
- `topopt_designs/`: topology optimization results (isotropic and orthotropic at 50%, 60%, 70%, and 80% density), with CSV files storing unit-cell designs and OBJ files storing the corresponding `6x6` tiled mesh

## Getting Started

Clone the repository and create the provided Conda environment for convenience and reproducibility.
Then install the package locally so the command-line tools can import `random_walk_microstructures` directly.

```bash
git clone https://github.com/samsilverman/random-walk-microstructures.git
cd random-walk-microstructures
conda env create -f environment.yml
conda activate random-walk-microstructures
pip install -e .
```

## Code

The codebase is organized into two parts:

- `src/random_walk_microstructures/`: the main Python package
- `tools/`: command-line entry points

| Script | Description | Image |
| - | - | - |
| `train.py` | Train the surrogate CNN model. | |
| `test.py` | Test the trained surrogate CNN model. | |
| `topopt.py` | Gradient-based topology optimization using the trained CNN surrogate. | ![topopt.py screenshot](https://github.com/samsilverman/random-walk-microstructures/blob/main/assets/images/topology_optimization.svg) |
| `visualize.py` | Visualize a CSV microstructure design. | ![visualize.py screenshot](https://github.com/samsilverman/random-walk-microstructures/blob/main/assets/images/visualize_design.svg) |
| `export_to_mesh.py` | Export a CSV microstructure design to an OBJ mesh. | ![export_to_mesh.py screenshot](https://github.com/samsilverman/random-walk-microstructures/blob/main/assets/images/export_to_mesh.svg) |

> [!NOTE]
> Each tool can be run from the repository root.
> For info on a specific tool/args run:
>
> ```bash
> python tools/[TOOL].py --help
> ```

## Maintainers

- [Sam Silverman](https://github.com/samsilverman/) - [sssilver@bu.edu](mailto:sssilver@bu.edu)

## Acknowledgements

The authors would like to thank Ruichen Liu for fabricating the optimized microstructure designs and Alec Ewe for running the compression tests on the fabricated samples.
This work was supported by a Focused Research Program from the Rafik B. Hariri Institute for Computing and Computational Science & Engineering at Boston University and by the National Science Foundation
(DMR-2323728).

## Citation

```bibtex
@inproceedings{Silverman:2025:RandomWalkMicrostructures,
author = {Silverman, Samuel and Balter, Dylan and Brown, Keith A. and Whiting, Emily},
title = {Random-Walk Microstructures for Differentiable Topology Optimization},
year = {2025},
isbn = {9798400720345},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3745778.3766645},
doi = {10.1145/3745778.3766645},
booktitle = {Proceedings of the ACM Symposium on Computational Fabrication},
articleno = {25},
numpages = {11},
keywords = {microstructures, random walks, inverse design, neural networks, homogenization},
location = {},
series = {SCF '25}
}
```

## License

Released under the [MIT License](https://github.com/samsilverman/random-walk-microstructures/blob/main/LICENSE).
