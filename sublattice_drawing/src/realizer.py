import copy
import networkx as nx

from typing import List, Tuple
from pysat.formula import CNF
from pysat.solvers import Solver
from fcapy.lattice import ConceptLattice

from fca.lattice import Lattice
from fca.context import Context

class SatRealizer:
    '''
    Compute a realizer for a given concept lattice using a SAT solver.

    Parameters
    ----------
    '''
    def __init__(self, lattice: ConceptLattice):
        # store lattice and concepts
        self.lattice = lattice
        self.graph = self.lattice.to_networkx()
        self.concepts = self.graph.nodes
        # store incomparable pairs
        self.incomparable_pairs = nx.complement(nx.transitive_closure(self.lattice.to_networkx()).to_undirected()).edges
        self.N_incomparable = 0
        if self.incomparable_pairs:
            self.N_incomparable = len(self.incomparable_pairs)
            # assign SAT variables to incomparable pairs
            # positive for (a,b), negative for (b,a)
            self.sat_variables = {}
            for i, (a, b) in enumerate(self.incomparable_pairs, start=1):
                self.sat_variables[(a, b)] = i
                self.sat_variables[(b, a)] = -i

    def realizer(self) -> Tuple[int, List[List[int]]]:
        '''
        Compute the realizer using the cadical195 SAT solver.

        Returns:
        --------
            dimension : int
                Order dimension of the concept lattice.
            realizer : List[List[int]]
                List of linear extensions forming the realizer.
        '''
        if self.N_incomparable == 0:
            return 1, [list(nx.topological_sort(self.graph))]

        # setup clauses for SAT solver
        self._setup_incomparability_clauses()

        dim = 1
        while True:
            solver = self._build_sat(dim)
            model = solver.get_model() if solver.solve() else None
            if model:
                break
            dim += 1
        
        return dim, self._realizer_from_sat_result(model, dim)

    def _setup_incomparability_clauses(self):
        '''
        Setup clauses to ensure transitivity among incomparable pairs.
        For each triple (a, b, c), ensure that if a < b and b < c, then a < c.
        '''
        self.incomparability_clauses = []
        for (a, c), ac_var in self.sat_variables.items():
            for b in self.concepts:
                # avoid trivial transitivity checks
                if b == a or b == c:
                    continue

                # get SAT variables for pairs
                ab_var = self.sat_variables.get((a, b))
                bc_var = self.sat_variables.get((b, c))

                # skip if (a,b) or (b,c) is not relevant for transitivity
                
                if ab_var is None and not b in nx.descendants(self.graph, a):
                    continue
                if bc_var is None and not c in nx.descendants(self.graph, b):
                    continue

                # Build clause: (~(a<b) OR ~(b<c) OR (a<c))
                clause = []
                if ab_var is not None:
                    clause.append(-ab_var)
                if bc_var is not None:
                    clause.append(-bc_var)
                clause.append(ac_var)

                self.incomparability_clauses.append(clause)

    def _build_sat(self, dim: int) -> Solver:
        '''
        Build the SAT solver instance for given dimension.

        Parameters
        ----------
        dim : int
            Dimension of the concept lattice (number of linear extensions).
        
        Returns
        -------
        solver : Solver
            SAT solver instance with added clauses.
        '''
        cnf = CNF()
        
        # Incomparability clauses for transitivity
        sign = lambda x: 1 if x > 0 else -1

        # Ensure independent variables for each linear extension
        for offset in [i * self.N_incomparable for i in range(dim)]:
            for clause in self.incomparability_clauses:
                cnf.append([var + sign(var) * offset for var in clause])

        for var in range(1, self.N_incomparable + 1):
            # positive clause: at least one extension has a < b
            cnf.append([var + i * self.N_incomparable for i in range(dim)])
            # negative clause: at least one extension has b < a
            cnf.append([-var - i * self.N_incomparable for i in range(dim)])

        return Solver(name="cadical195", bootstrap_with=cnf)

    def _realizer_from_sat_result(self, model, dim: int) -> List[List[int]]:
        '''
        Construct the realizer from the SAT solver result.

        Parameters
        ----------
        result : Dict[int, bool]
            Mapping from SAT variable to boolean value indicating the order.
        dim : int
            Dimension of the concept lattice (number of linear extensions).

        Returns
        -------
        realizer : List[List[int]]
            List of linear extensions forming the realizer.
        '''
        realizer = []

        for i in range(dim):
            le = copy.deepcopy(self.graph)

            # offset for i-th linear extension
            offset = i * self.N_incomparable

            # reach total order by adding edges between incomparable pairs according to SAT result
            for var_i, (a, b) in enumerate(self.incomparable_pairs, start=1):
                if (var_i + offset) in model:
                    le.add_edge(a, b)
                else:
                    le.add_edge(b, a)

            # topological sort gives the linear extension
            realizer.append(list(nx.topological_sort(le)))

        return realizer