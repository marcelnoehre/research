import itertools
import pandas as pd

from typing import Set, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def object_concept(
        context: FormalContext,
        g: str
    ) -> Tuple[Set[str], Set[str]]:
    '''
    Compute the object concept of g.

    Parameters
    ----------
    formal_context : FormalContext
        The formal context
    g : str
        The object

    Returns
    -------
    concept: Tuple[Set[str], Set[str]]
        The object concept
    '''
    return (set(context.extension(context.intention({g}))), set(context.intention({g})))

def attribute_concept(
        context: FormalContext,
        m: str
    ) -> Tuple[Set[str], Set[str]]:
    '''
    Compute the attribute concept of m.

    Parameters
    ----------
    context : FormalContext
        The formal context
    m : str
        The attribute

    Returns
    -------
    concept: Tuple[Set[str], Set[str]]
        The attribute concept
    '''
    return (set(context.extension({m})), set(context.intention(context.extension({m}))))

def object_closure(
        context: FormalContext,
        objects: Set[str]
    ) -> Set[str]:
    '''
    Compute the closure of a set of objects.

    Parameters
    ----------
    context : FormalContext
        The formal context
    objects : Set[str]
        The set of objects

    Returns
    -------
    closure: Set[str]
        The double-primed set of objects
    '''
    return set(context.extension(context.intention(objects)))

def attribute_closure(
        context: FormalContext,
        attributes: Set[str]
    ) -> Set[str]:
    '''
    Compute the closure of a set of attributes.

    Parameters
    ----------
    context : FormalContext
        The formal context
    attributes : Set[str]
        The set of attributes

    Returns
    -------
    closure: Set[str]
        The double-primed set of attributes
    '''
    return set(context.intention(context.extension(attributes)))

def reduce_context(
        context: FormalContext
    ) -> FormalContext:
    '''
    Reduce the formal context by keeping only join-irreducible objects and meet-irreducible
    attributes.

    Parameters
    ----------
    context : FormalContext
        The formal context to be reduced

    Returns
    -------
    reduced_context: FormalContext
        The reduced formal context
    '''
    lattice = ConceptLattice.from_context(context)
    join_irreducibles = []
    meet_irreducibles = []

    for c in lattice.to_networkx().nodes:
        
        # join-irreducible
        if len(lattice.children(c)) == 1:
            join_irreducibles.append(list(lattice.get_concept_new_extent(c))[0])

        # meet-irreducible
        if len(lattice.parents(c)) == 1:
            meet_irreducibles.append(list(lattice.get_concept_new_intent(c))[0])

    # already reduced
    if len(join_irreducibles) == context.n_objects and len(meet_irreducibles) == context.n_attributes:
        return context
    
    # create reduced context
    else:
        df = pd.DataFrame(0, index=join_irreducibles, columns=meet_irreducibles)
        for g, m in itertools.product(join_irreducibles, meet_irreducibles):
            if m in context.intention([g]):
                df.loc[g, m] = 1
        
        return FormalContext(data=df.values.astype(bool).tolist(), object_names=join_irreducibles, attribute_names=meet_irreducibles)
