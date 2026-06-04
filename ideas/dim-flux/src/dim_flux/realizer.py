import numpy as np
from pathlib import Path

from odis import FormalContext
from fcapy.lattice import ConceptLattice

from src.utils.variables import Variables
from src.fca.lattice import compute_lectic_order
from src.dim_flux.projection import Projection
from src.dim_flux.lgs import LinearEquationSolver

class Realizer():
    '''
    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''
    def __init__(self,
            variables: Variables
        ):
        self.vars = variables
        self.context = variables.context
        self.lattice = ConceptLattice.from_context(variables.context)
        
        self.coordinates = self.two_dimensional_extension()
        self.vars.coordinates = self.coordinates
        projection = Projection(self.vars)
        self.vars.coordinates = projection.coordinates
        self._derive_base_vectors()

    def two_dimensional_extension(self):
        '''
        Compute the two-dimensional extension of the lattice.

        Returns
        -------
        coordinates : Dict[int, List]
            Original DimDraw coordinates.
        '''
        if self.vars.cxt.endswith('.cxt'):
            cxt_path = Path(self.vars.cxt).resolve()
        else:
            cxt_path = Path(f'data/{self.vars.cxt}.cxt').resolve()

        ctx = FormalContext.from_file(str(cxt_path))
        drawing = ctx.draw("dimdraw")
        self.lectic_order = compute_lectic_order(self.vars)
        self.coordinates = {
            c: (np.array([drawing.nodes[i].x, drawing.nodes[i].y]) * -1 * np.array([np.sqrt(2), 1/np.sqrt(2)])).tolist()
            for i, c in enumerate(self.lectic_order)
        }
        pos = [f'{self.coordinates[c][0]} {self.coordinates[c][1]}' for c in self.vars.concepts]
        with open(f'evaluation/positions/dim_draw/{self.vars.name}.pos', 'w', encoding='utf-8') as f:
            f.write('\n'.join(pos))
        return self.coordinates

    def _derive_base_vectors(self):
        '''
        Derive base vectors by solving the system of linear equations
        '''
        lgs = LinearEquationSolver(self.vars, self.coordinates)
        success, vector_vars = lgs.solve_linear_equations()
        if success:
            self.base_vectors = dict({})
            for v in self.vars.elements:
                if v in self.vars.G:
                    self.base_vectors[v] = np.array([vector_vars[f'x_{v}'], vector_vars[f'y_{v}']])
                else:
                    self.base_vectors[v] = np.array([-vector_vars[f'x_{v}'], -vector_vars[f'y_{v}']])