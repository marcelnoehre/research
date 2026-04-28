import networkx as nx

from typing import Tuple, Set, Dict
from collections import deque 
from fcapy.lattice import ConceptLattice

def cover_relations(concept_lattice: ConceptLattice) -> Set[Tuple[int, int]]:
    '''
    Get the cover relations of a concept lattice.

    Parameters
    ----------
    concept_lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    cover_relations : Set[Tuple[int, int]]
        A set of tuples representing the cover relations of the lattice.
    '''
    return set(nx.transitive_reduction(concept_lattice.to_networkx()).edges)

def all_extents(
        lattice: ConceptLattice
    ) -> Dict[int, Set[str]]:
    '''
    Compute the extents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    extents: Dict[int, Set[str]]
        A dictionary mapping concept IDs to their full extents
    '''
    seen = set({})
    queue = deque({len(lattice.to_networkx().nodes)-1})
    extents = dict({})

    while queue:
        concept = queue.popleft()
        
        # all childrens processed?
        if lattice.children(concept) <= extents.keys():
            # A_new \cup A_children
            extents[concept] = lattice.get_concept_new_extent(concept).union(
                *(extents[c] for c in lattice.children(concept))
            )

            # add parents to queue
            seen.update(lattice.parents(concept) - extents.keys())
            queue.extend(lattice.parents(concept) - extents.keys())

        # readd to queue
        else:
            queue.append(concept)

    return extents

def all_intents(
        lattice: ConceptLattice
    ) -> Dict[int, Set[str]]:
    '''
    Compute the intents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    intents: Dict[int, Set[str]]
        A dictionary mapping concept IDs to their full intents
    '''
    seen = set({})
    queue = deque({0})
    intents = dict({})

    while queue:
        concept = queue.popleft()
        
        # all parents processed?
        if lattice.parents(concept) <= intents.keys():
            # B_new \cup B_parents
            intents[concept] = lattice.get_concept_new_intent(concept).union(
                *(intents[p] for p in lattice.parents(concept))
            )

            # add children to queue
            seen.update(lattice.children(concept) - intents.keys())
            queue.extend(lattice.children(concept) - intents.keys())

        # readd to queue
        else:
            queue.append(concept)

    return intents

def incomparability_graph(lattice: ConceptLattice) -> nx.Graph:
    '''
    Get the incomparability graph of a concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    
    Returns
    -------
    incomparability_graph : nx.Graph
        The incomparability graph of the lattice.
    '''
    return nx.complement(nx.transitive_closure(lattice.to_networkx()).to_undirected())

def _lectically_smaller(vars, intent_a: set, intent_b: set) -> bool:
    '''
    Check if concept A is lectically smaller than concept B.
    
    A <L B iff there exists an attribute m in M such that:
        - m is the smallest attribute in (intent_b - intent_a) \cup (intent_a - intent_b)
        - m \in intent_b (B contains it, A does not)

    Parameters
    ----------
    intent_a : set
        Intent of concept A
    intent_b : set
        Intent of concept B

    Returns
    -------
    bool
        True if A is lectically smaller than B
    '''
    if intent_a == intent_b:
        return False
    
    for m in vars.attributes:
        if m in intent_a and m in intent_b:
            continue
        if m not in intent_a and m not in intent_b:
            continue
        # m is the smallest differing element
        # A <L B iff A does NOT contain m
        return m not in intent_a
    
    return False

def compute_lectic_order(vars) -> list:
    '''
    Sort all concepts by the lectic order on their intents.

    Returns
    -------
    list
        Concept IDs sorted lectically
    '''
    
    concepts = list(vars.concepts)

    for i in range(1, len(concepts)):
        key = concepts[i]
        key_intent = vars.intents[key]
        j = i - 1
        while j >= 0 and _lectically_smaller(vars, key_intent, vars.intents[concepts[j]]):
            concepts[j + 1] = concepts[j]
            j -= 1
        concepts[j + 1] = key

    return concepts