import numpy as np

from collections import deque 

from src.utils.variables import Variables

class InitLayout():
    '''
    Compute the initial coordinates for a concept lattice based on the Sup-Inf Distance.

    Parameters
    ----------
    vars : Variables
        The container class holding the data
    base_vectors : Dict[str, np.ndarray]
        Mapping of objects and attributes to their assigned base vectors
    coordinates : Dict[int, np.ndarray]
        Mapping of concept IDs to their final computed 2D coordinates
    '''

    def __init__(self, variables: Variables):
        '''
        Initialize the layout process and compute base vectors and 
        concept coordinates.

        Parameters
        ----------
        variables : Variables
            The container class holding context data and layout parameters
        '''
        self.vars = variables
        self.base_vectors = dict({})

        # object vectors
        self._init_atoms()
        self._object_chain_decomposition()
        
        # attribute vectors
        self._init_coatoms()
        self._attribute_chain_decomposition()

        # initial layout
        self._derive_coordinates()

    def _init_atoms(self):
        '''
        Assign base vectors to atoms (join-irreducibles above the bottom) using a parabolic distribution.
        '''
        ordered_atoms = sorted(self.vars.atoms, key=lambda a: self.vars.scalars[self.vars.element_map[a]])
        for i, g in enumerate(ordered_atoms, start=1):
            x = round(1.8 * i - 0.9 * (len(self.vars.atoms) + 1), 1)
            y = 0.09 * x**2 + 1.75
            self.base_vectors[g] = np.array([x, y])

    def _init_coatoms(self):
        '''
        Assign base vectors to coatoms (meet-irreducibles below the top) using an inverted parabolic distribution.
        '''
        ordered_coatoms = sorted(self.vars.coatoms, key=lambda c: self.vars.scalars[self.vars.element_map[c]])
        for i, m in enumerate(ordered_coatoms, start=1):
            x = -round(1.8 * i - 0.9 * (len(self.vars.coatoms) + 1), 1)
            y = -(0.09 * x**2 + 1.75)
            self.base_vectors[m] = np.array([x, y])

    def _object_chain_decomposition(self):
        '''
        Compute base vectors for the remaining objects by decomposing the poset and averaging lower neighbor vectors.
        '''
        remaining_objects = self.vars.G - set(self.vars.atoms)
        queue = deque(remaining_objects)
        
        # precompute lower neighbors
        lower = {
            g: {g_i for g_i, cl in self.vars.object_closures.items() if cl < self.vars.object_closures[g]}
            for g in remaining_objects
        }

        while queue:
            g = queue.popleft()

            # \bot concept
            if g in self.vars.lattice.get_concept_new_extent(self.vars.N_c - 1):
                self.base_vectors[g] = np.zeros(2)
            
            # all lower irreducible objects already processed
            elif all(low in self.base_vectors for low in lower[g]):
                # handle all objects with the same lower neighbors
                same_lower = [k for k, v in lower.items() if v == lower[g]]
                # sort based on scalar order derived from d_SI
                ordered_same_lower = sorted(same_lower, key=lambda a: self.vars.scalars[self.vars.object_map[a]])
                # arithmetic mean of lower neighbors
                mean = sum([self.base_vectors[low] for low in lower[g]]) / len(lower[g])
                for g_i in ordered_same_lower:
                    # offset based on scalar order
                    delta_i = 1e-3 * self.vars.element_map[g_i]
                    self.base_vectors[g_i] = mean + delta_i
                # drop batch of processed objects
                queue = deque([g for g in queue if g not in same_lower])

            else:
                queue.append(g)

    def _attribute_chain_decomposition(self):
        '''
        Compute base vectors for the remaining attributes by decomposing the poset and averaging upper neighbor vectors.
        '''
        remaining_attributes = self.vars.M - set(self.vars.coatoms)
        queue = deque(remaining_attributes)

        # precompute upper neighbors
        upper = {
            m: self.vars.attribute_closures[m] - {m}
            for m in remaining_attributes
        }

        while queue:
            m = queue.popleft()

            # \top concept
            if m in self.vars.lattice.get_concept_new_intent(0):
                self.base_vectors[m] = np.zeros(2)

            # all upper attributes already processed
            elif all(up in self.base_vectors for up in upper[m]):
                # handle all attributes with the same upper neighbors
                same_upper = [k for k, v in upper.items() if v == upper[m]]
                # sort based on scalar order derived from d_SI
                ordered_same_upper = [
                    self.vars.attributes[m] 
                    for m in self.vars.order 
                    if m in [self.vars.attribute_map[su] for su in same_upper]
                ]
                # arithmetic mean of upper neighbors
                mean = sum([self.base_vectors[up] for up in upper[m]]) / len(upper[m])
                for m in ordered_same_upper:
                    # offset based on scalar order
                    delta_i = 1e-3 * self.vars.element_map[m]
                    self.base_vectors[m] = mean + delta_i
                # drop batch of processed attributes
                queue = deque([m for m in queue if m not in same_upper])

            else:
                queue.append(m)

    def _derive_coordinates(self):
        '''
        Calculate the final 2D coordinates for each concept as the vector sum of its extent and intent base vectors.
        '''
        base_vectors = np.array([self.base_vectors[v] for v in self.vars.elements])
        self.coordinates = {}
        for concept in self.vars.concepts:
            concept_vectors = np.array([
                self.vars.element_map[v] 
                for v in (self.vars.extents[concept]) | self.vars.intents[concept]
            ])

            # (0, 0) if concept has no irreducible objects and no attributes
            if not concept_vectors.size:
                self.coordinates[concept] = np.zeros(2)
            
            self.coordinates[concept] = np.sum(base_vectors[concept_vectors], axis=0)
