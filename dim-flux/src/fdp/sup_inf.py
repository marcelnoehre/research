import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from src.fca.context import *
from src.utils.variables import Variables

class SupInfGraph():
    '''
    Compute the Supremum-Infimum (SI) graph and distances to determine the 
    linear ordering of irreducible elements for the lattice layout.

    Parameters
    ----------
    vars : Variables
        The container holding context data and layout parameters
    dsi_matrix : np.ndarray
        A symmetric matrix storing computed d_SI distances between all elements
    d_si_points : np.ndarray
        The 2D coordinates of elements after spring model optimization and rotation
    n_1 : np.ndarray
        The start point of the longest path (f_max) in the SI graph
    n_2 : np.ndarray
        The end point of the longest path (f_max) in the SI graph
    scalars : np.ndarray
        Projection of SI points onto the f_max vector for linear ordering
    order : np.ndarray
        The sorted indices of elements based on their scalar projections
    '''
    def __init__(self, variables: Variables):
        '''
        Initialize the SI graph and compute the distances and layout.

        Parameters
        ----------
        variables : Variables
            The container class holding the data
        '''
        self.vars = variables
        self.dsi_matrix = np.zeros((self.vars.N_e, self.vars.N_e))
        
        self._sup_inf_distance()
        self._sup_inf_graph()

    def _sup_inf_distance(self):
        '''
        Compute the Supremum-Infimum distance for all combinations of 
        irreducible objects and attributes using concept intersections 
        and closures.
        '''
        S = {
            i: (
                # ({g}'', {g}')
                object_concept(self.vars.context, self.vars.elements[i])
                if i < self.vars.N_g
                # ({m}', {m}'')
                else attribute_concept(self.vars.context, self.vars.elements[i])
            )
            for i in range(self.vars.N_e)
        }

        for i, j in combinations(range(self.vars.N_e), 2):
            # g_i, g_j
            if j < self.vars.N_g:
                # g_i || g_j 
                if not(S[i][1] <= S[j][1] or S[j][1] <= S[i][1]):
                    # g_i'' \cap g_j''
                    inf = S[i][0] & S[j][0]
                    # (g_i \cup g_j)''
                    sup = object_closure(self.vars.context, (S[i][0] | S[j][0]))
                    # |Sup_g| - |Inf_g| - 1
                    self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = len(sup) - len(inf) - 1
                
            # m_i, m_j
            elif i >= self.vars.N_g:
                # m_i || m_j 
                if not(S[i][0] <= S[j][0] or S[j][0] <= S[i][0]):
                    # m_i'' \cap m_j''
                    sup = S[i][1] & S[j][1]
                    # (m_i \cup m_j)''
                    inf = attribute_closure(self.vars.context, S[i][1] | S[j][1])
                    # |Sup_m| |Inf_m| - 1
                    self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = len(inf) - len(sup) - 1
            
            # g_i, m_j
            else:
                # g_i || m_j
                if not(self.vars.elements[i] in S[j][0]):
                    # g_i'' \cap g_j''
                    inf_g = S[i][0] & S[j][0]
                    # m_i'' \cap m_j''
                    sup_m = S[i][1] & S[j][1]
                    # (g_i \cup g_j)''
                    sup_g = object_closure(self.vars.context, S[i][0] | S[j][0])
                    # (m_i \cup m_j)''
                    inf_m = attribute_closure(self.vars.context, S[i][1] | S[j][1])
                    # (|Sup_g| - |Sup_m|) (|Inf_g| - |Inf_m|) - 1
                    self.dsi_matrix[i, j] = self.dsi_matrix[j, i] = (len(inf_g) - len(sup_m)) - (len(sup_g) - len(inf_m)) - 1

    def _sup_inf_graph(self):
        '''
        Construct the SI graph by solving a spring model, finding the 
        maximal axis, and rotating the system to align with the horizontal axis.
        '''
        # intial spring layout
        spring_pts = self._solve_spring_model()

        # longest path
        dists = squareform(pdist(spring_pts))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)

        # n_1 lies left of n_2
        if spring_pts[i, 0] < spring_pts[j, 0]:
            n_1, n_2 = spring_pts[i], spring_pts[j]
        else:
            n_1, n_2 = spring_pts[j], spring_pts[i]

        # f_max
        f_max = n_2 - n_1
        angle = -np.arctan2(f_max[1], f_max[0])

        # rotate f_max horizontal
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        self.d_si_points = spring_pts @ rot.T
        self.n_1 = n_1 @ rot.T
        self.n_2 = n_2 @ rot.T

        # derive order of vectors
        self.scalars = np.dot(spring_pts - n_1, n_2 - n_1)
        self.order = np.argsort(self.scalars)
        
    def _solve_spring_model(self) -> np.ndarray:
        '''
        Solve the force-directed layout for the SI graph by minimizing 
        the squared difference between Euclidean and SI distances.

        Returns
        -------
        optimized_positions: np.ndarray
            The 2D positions of irreducible elements in the spring model
        '''
        def e_si(flat_positions) -> Tuple[float, np.ndarray]:
            '''
            Compute the energy of the sup-inf system and the analytical gradient.

            Parameters
            ----------
            flat_positions : np.array
                A 1D flattened array of object and attribute vectors
            
            Returns
            -------
            energy: float
                The total system energy
            gradient: np.ndarray
                The flattened gradient vector
            '''
            positions = flat_positions.reshape(-1, 2)
            gradients = np.zeros_like(positions)

            # E_SI
            e_si = 0.0
            for i, j in combinations(range(self.vars.N_e), 2):
                n_i = positions[i]
                n_j = positions[j]

                # (|n_i - n_j| - d_SI(n_i, n_j))^2
                e_si += (np.linalg.norm(n_i - n_j) - self.dsi_matrix[i, j])**2

            # F_SI(n_i)
            for i in range(self.vars.N_e):
                for j in range(self.vars.N_e):
                    if i == j:
                        continue
                    n_i = positions[i]
                    n_j = positions[j]

                    # -2 * ((|n_i - n_j| - d_SI(n_i, n_j)) / |n_i - n_j|) * (n_i - n_j)
                    gradients[i] += -2 * ((np.linalg.norm(n_i - n_j) - self.dsi_matrix[i, j]) / np.maximum(np.linalg.norm(n_i - n_j), 1e-9)) * (n_i - n_j)
                    
            # conjugate gradient (expects a negative gradient)
            return e_si, (gradients*-1).flatten()

        # intialize points at unit circle
        initial_pts = np.zeros((self.vars.N_e, 2))
        for i in range(self.vars.N_e):
            phi = 2 * np.pi * i / self.vars.N_e
            initial_pts[i] = [np.cos(phi), np.sin(phi)]

        res = minimize(
            e_si,
            initial_pts.flatten(),
            method='CG',
            jac=True
        )
        return res.x.reshape(-1, 2)
