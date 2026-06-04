import numpy as np

from src.utils.variables import Variables

class SnapToGrid:
    '''
    Snap the base vectors to the nearest grid point
    '''
    def __init__(self,
            variables: Variables
        ):
        self.vars = variables
        self.base_vectors = {k: np.round(v) for k, v in self.vars.base_vectors.items()}
        self.coordinates = {}
        for c in self.vars.concepts:
            self.coordinates[c] = self._concept_pos(c, np.array([self.vars.base_vectors[v] for v in self.vars.elements]))

    def _concept_pos(self, concept, vectors) -> np.ndarray:
        '''
        Compute concept position based on present vectors.

        Parameters
        ----------
        concept : int
            Concept to compute the position for
        vectors : Dict[int, np.array]
            Dictionary assigning vectors to objects and attributes  

        Returns
        -------
        position : np.array
            Position of the concept
        '''
        base_vectors = np.array([self.vars.element_map[v] for v in (self.vars.extents[concept] | self.vars.intents[concept])])

        # (0, 0) if concept has no objects and no attributes
        if not base_vectors.size:
            return np.zeros(2)
        
        return np.sum(vectors[base_vectors], axis=0)