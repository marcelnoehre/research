import numpy as np

from typing import Optional
from dataclasses import dataclass
from fcapy.lattice import ConceptLattice

from src.fca.context import *
from src.fca.lattice import *
from src.utils.parser import decode_cxt

@dataclass
class Args:
    plot_si_graph: bool = False
    si_graph_annotations: bool = False
    plot_initial_layout: bool = False
    initial_layout_annotations: bool = False
    plot_optimized_layout: bool = False
    optimized_layout_annotations: bool = False
    plot_individual_forces: bool = False
    plot_combined_forces: bool = False
    plot_gradients: bool = False
    plot_origin: bool = False

class Variables():
    '''
    A container class to store the formal context, concept lattice, 
    and parameters required for lattice drawing and force-directed layouts.

    Parameters
    ----------
    cxt : str
        The filename or path of the formal context
    context : FormalContext
        The formal context decoded from the input file
    lattice : ConceptLattice
        The concept lattice derived from the context
    args : Args
        Configuration object for visualization settings
    objects : List[str]
        List of object names in the context
    object_map : Dict[str, int]
        Mapping of object names to their integer indices
    object_closures : Dict[str, Set[str]]
        Mapping of objects to their closure sets
    N_g : int
        The number of objects in the context
    G : Set[str]
        The set of all object names
    attributes : List[str]
        List of attribute names in the context
    attribute_map : Dict[str, int]
        Mapping of attribute names to their integer indices
    attribute_closures : Dict[str, Set[str]]
        Mapping of attributes to their closure sets
    N_m : int
        The number of attributes in the context
    M : Set[str]
        The set of all attribute names
    elements : List[str]
        Concatenated list of objects and attributes
    element_map : Dict[str, int]
        Mapping of element names to their integer indices
    N_e : int
        Total number of elements (objects + attributes)
    E : Set[str]
        The set of all elements
    concepts : List[int]
        List of concept IDs from the lattice
    N_c : int
        Total number of concepts in the lattice
    extents : Dict[int, Set[str]]
        Mapping of concept IDs to their extents
    intents : Dict[int, Set[str]]
        Mapping of concept IDs to their intents
    atoms : List[str]
        Objects belonging to concepts directly above the bottom concept
    coatoms : List[str]
        Attributes belonging to concepts directly below the top concept
    w_rep : float
        Repulsive force weight
    w_att : float
        Attractive force weight
    w_grav : float
        Gravitational force weight
    order : List
        The processing order for layout optimization
    scalars : np.ndarray
        Array of scalar values associated with each vector
    d_si_points : List
        Points used for sup inf graph calculations
    n_1 : int
        Left point of f_max
    n_2 : int
        Right point of f_max
    base_vectors : Dict
        Dictionary storing the base vectors for elements
    coordinates : Dict
        Dictionary mapping concepts to their computed coordinates
    final_forces : Dict
        Dictionary storing the resultant forces after optimization
    '''

    def __init__(self, cxt: str, args: Optional[Dict[str, bool]]):

        self.cxt = cxt
        if cxt.endswith('.cxt'):
            self.context = decode_cxt(cxt)
        else:
            self.context = decode_cxt(f'data/{cxt}.cxt')
        self.lattice = ConceptLattice.from_context(self.context)
        self.args: Args = Args(**(args or {}))

        # objects
        self.objects = self.context.object_names
        self.object_map = self.context._object_names_i_map
        self.object_closures = {
            g: object_closure(self.context, {g})
            for g in self.objects
        }
        self.N_g = self.context.n_objects
        self.G = set(self.objects)
        
        # attributes
        self.attributes = self.context.attribute_names
        self.attribute_map = self.context._attribute_names_i_map
        self.attribute_closures = {
            m: attribute_closure(self.context, {m})
            for m in self.attributes
        }
        self.N_m = self.context.n_attributes
        self.M = set(self.attributes)

        # elements
        self.elements = self.objects + self.attributes
        self.element_map = {
            v: i
            for i, v in enumerate(self.elements)
        }
        self.N_e = self.N_g + self.N_m
        self.E = set(self.elements)

        # concepts
        self.concepts = self.lattice.to_networkx().nodes
        self.N_c = len(self.concepts)
        self.extents = all_extents(self.lattice)
        self.intents = all_intents(self.lattice)
        self.atoms = [
            next(iter(self.lattice.get_concept_new_extent(c)))
            for c in self.lattice.parents(self.N_c - 1)
        ]
        self.coatoms = [
            next(iter(self.lattice.get_concept_new_intent(c)))
            for c in self.lattice.children(0)
        ]

        # weights
        self.w_rep = 50.0
        self.w_att = 1.0
        self.w_grav = 30.0

        # global variables
        self.order = []
        self.scalars = np.zeros(self.N_e)        
        self.d_si_points = []
        self.n_1 = 0
        self.n_2 = 0
        self.base_vectors = dict({})
        self.coordinates = dict({})
        self.final_forces = dict({})
