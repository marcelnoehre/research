# DimFlux: Force-Directed Doubly-Additive Line Diagrams
This repository implements a force-directed placement approach for doubly-additive line drawings.

## References
This project is based on the theory and methods described in the following works.

### Primary Reference
```bash
@misc{nöhre2026dimfluxforcedirectedadditiveline,
  title = {DimFlux: Force-Directed Additive Line Diagrams},
  author = {Marcel Nöhre and Dominik Dürrschnabel and Bernhard Ganter and Gerd Stumme},
  year = {2026},
  eprint={2603.16366},
  archivePrefix={arXiv},
  primaryClass={cs.CG},
  url={https://arxiv.org/abs/2603.16366}
}
```

### Supplementary Material
```bash
@misc{nohre_2026_18936106,
  author = {Nöhre, Marcel and Dürrschnabel, Dominik and Ganter, Bernhard and Stumme, Gerd},
  title = {A Visual Benchmark of DimFlux: Comparison of Line Diagrams for Concept Lattices},
  month = mar,
  year = 2026,
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18936106},
  url = {https://doi.org/10.5281/zenodo.18936106}
}
```

*If you use this code in your work, please cite the above references.*


## Usage
Clone the repository and initialize the environment. uv will automatically create a virtual
environment and sync dependencies based on the pyproject.toml. `uv` will automatically create a
virtual environment.

```bash
uv sync
```

Execute the layout generation using uv run to ensure the correct environment context.
```bash
uv run python main.py
```

The script will prompt you for a lattice number corresponding to the `.cxt` files located in the
`/data` directory.

## Configuration
| Category | Parameter | Description |
| :--- | :--- | :--- |
| **Drawings** | `plot_si_graph` | Visualize the Supremum-Infimum graph ($f_{max}$ is highlighted) |
| | `plot_initial_layout` | Visualize the initial layout of the concept lattice |
| | `plot_optimized_layout` | Visualize the final optimized layout of the concept lattice |
| **Annotations** | `si_graph_annotations` | Display annotations in the SI-Graph |
| | `initial_layout_annotations` | Display annotations in the initial drawing |
| | `optimized_layout_annotations` | Display annotations in the final optimized drawing |
| **Forces** | `plot_individual_forces` | Display force vectors for objects and attributes respectively |
| | `plot_combined_forces` | Display combined force vectors |
| | `plot_gradients` | Display gradient vectors at concepts |
| **Dev** | `plot_origin` | Display the origin $(0,0)$ |

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
