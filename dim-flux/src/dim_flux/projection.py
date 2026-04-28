import numpy as np

from src.fca.lattice import compute_lectic_order
from src.utils.variables import Variables

class Projection():
    def __init__(self,
            variables: Variables
        ):
        '''
        Project a line diagram into the space of additive line diagrams.

        Parameters
        ----------
        variables: Variables
            The storage of variables
        '''
        self.vars = variables
        self.lectic_order = compute_lectic_order(self.vars)
        self._set_representation()
        self._orthonormal_basis()
        self._additive_coordinates()

    def _set_representation(self):
        '''
        Compute the Set Representation Matrix (SRM).

        Each row corresponds to a concept in lectic order. Each column
        corresponds to an element (object or attribute) in the order defined
        by objects + attributes.

        An element is flagged (1) if it is:
            - an object belonging to the extent of c, or
            - an attribute NOT belonging to the intent of c.
        '''
        self.srm = np.array([
            [
                1 if e in (self.vars.extents[c] | (self.vars.M - self.vars.intents[c])) else 0
                for e in self.vars.elements
            ]
            for c in self.lectic_order
        ])

    def _orthonormal_basis(self):
        '''
        Compute an orthonormal basis for the column space of the SRM.

        Applies Gram-Schmidt orthogonalisation column-wise to self.srm,
        discarding linearly dependent columns (norm below 1e-5).

        Sets
        ----
        self.basis : np.ndarray of shape (N_c, k)
            Matrix whose columns form an orthonormal basis for the column
            space of self.srm, where k <= N_e is the rank of the SRM.
        '''
        self.basis = np.zeros((len(self.srm),0))
        for srm_col_n in range(self.srm.shape[1]):
            projection = np.zeros((len(self.srm),1))
            srm_col = self.srm[:,srm_col_n:srm_col_n+1]
            if self.basis.shape[1] > 0:
                for bcolnum in range(self.basis.shape[1]):
                    bcol = self.basis[:,bcolnum:bcolnum+1]
                    projection += np.dot(srm_col.T,bcol)*bcol
            newcol = srm_col - projection
            norm = np.linalg.norm(newcol)
            if norm > 1.e-05:
                newcol = newcol/norm
                self.basis = np.column_stack((self.basis,newcol))

    def _additive_coordinates(self):
        '''
        Compute the closest additive placement to the current coordinates.
        '''
        # (N_c, 2) position matrix in lectic order
        xy = np.array([self.vars.coordinates[c] for c in self.lectic_order])
        # project onto column space of SRM
        self.projected_xy = self.basis @ (self.basis.T @ xy)
        self.coordinates = {}
        for i, c in enumerate(self.lectic_order):
            self.coordinates[c] = self.projected_xy[i]