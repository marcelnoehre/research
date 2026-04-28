import numpy as np

from typing import Dict
from collections import deque
from src.utils.variables import Variables
from sympy import symbols, Eq, solve, sympify, linear_eq_to_matrix

class LinearEquationSolver:
    '''
    Solve a system of linear equations derived from a concept lattice structure.

    Parameters
    ----------
    variables : Variables
        The underlying concept lattice.
    coordinates : Dict[int, np.ndarray]
        A mapping from lattice node indices to their coordinates.
    '''
    def __init__(self,
            variables: Variables,
            coordinates: Dict[int, np.ndarray]
        ):
        self.vars = variables
        self.coordinates = coordinates
        self.dimensions = ['x', 'y']

        self.variables=[f'{dim}_{v}' for dim in self.dimensions for v in self.vars.objects + self.vars.attributes]
        self.symbols = symbols(' '.join(self.variables))

        self._construct_equations()
        self.solution = solve(self.equations, self.symbols, dict=True)
        if not self.solution:
            self._solve_approximate()

    def _solve_approximate(self):
        A, b = linear_eq_to_matrix(self.equations, self.symbols)
        sol_matrix = A.pinv() * b
        self.solution = [{self.symbols[i]: sol_matrix[i] for i in range(len(self.symbols))}]

    def _construct_equations(self):
        self.equations = []
        
        for c in self.vars.concepts:
            elements = self.vars.extents[c] | (self.vars.M - self.vars.intents[c])
            for i, dim in enumerate(self.dimensions):
                l, r = tuple(((' + '.join(f'{dim}_{v}' for v in elements) if elements else '0'), f'{self.coordinates[c][i]}'))
                eq = Eq(sympify(l), sympify(r))
                if eq != True:
                    self.equations.append(eq)
            
    def _solve(self, dim: str, node: int, expected: int):
        '''
        Solve the equation for a specific node and dimension.
        
        Parameters
        ----------
        dim : str
            The dimension for which to solve the equation.
        node : int
            The lattice node index.
        expected : int
            The expected value for the equation.
        '''
        eq = [] # construct equation to solve

        for var in self.vars.extents[node] | (self.vars.M - self.vars.intents[node]):
            var = f'{dim}_{var}'

            # if variable value is already known, insert it directly
            if var in self.vector_variables:
                eq.append(str(self.vector_variables[var]))

            # if variable is free, insert symbol
            elif symbols(var) in self.free_vars:
                eq.append(str(var))

            else:
                # construct equation
                sub_eq = str(self.solution[0][symbols(var)]) 

                # insert values of already known variables
                for k, v in self.vector_variables.items():
                    sub_eq = sub_eq.replace(str(k), str(v))
                
                # solve if no variables are left in the sub-equation
                if not any(var in sub_eq for var in self.variables):
                    self.vector_variables[var] = float(eval(sub_eq, {"__builtins__": None}))
                    sub_eq = str(self.vector_variables[var])

                eq.append(f'({sub_eq})')

        # final equation to solve
        eq = ' + '.join(eq)

        # if the equation contains variables, solve it
        if any(var in eq for var in self.variables):
            # define sympy variables and equations
            expr_vars = [v for v in self.variables if eq.find(v) != -1]
            expr_symbols = symbols(' '.join(expr_vars))
            solution = solve(Eq(sympify(eq), expected), expr_symbols, dict=True)
            
            # if a solution is found, update the variable values
            if solution:
                for k, v in solution[0].items():
                    self.vector_variables[str(k)] = float(v)
            
            # if no solution is found the variables cancel out -> assign 0 to all variables
            else:
                for var in expr_vars:
                    if str(var) not in list(self.vector_variables.keys()):
                        self.vector_variables[str(var)] = 0

    def solve_linear_equations(self):
        '''
        Solve the system of linear equations.

        Returns
        -------
        success : bool
            True if the equations were successfully solved, False otherwise.
        vector_variables : Dict[str, int]
            A mapping from variable names to their solved integer values.
        '''
        if not self.solution:
            return False, None
        
        # identify free variables that can take any value
        self.free_vars = [v for v in self.symbols if v not in self.solution[0]]

        # extract already fixed variable values
        self.vector_variables = {}
        for k, v in self.solution[0].items():
            try:
                self.vector_variables[str(k)] = float(v)
            except TypeError:
                continue

        visited = set()
        queue = deque([self.vars.N_c - 1])

        while queue:
            node = queue.popleft()

            # solve dimensions separately
            for i, dim in enumerate(self.dimensions):
                while not all(f'{dim}_{var}' in self.vector_variables.keys()
                    for var in self.vars.extents[node] | (self.vars.M - self.vars.intents[node])
                ):              
                    self._solve(dim, node, self.coordinates[node][i])
            
            # add parents if not already processed or in queue
            visited.add(node)
            for p in self.vars.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

        return True, self.vector_variables