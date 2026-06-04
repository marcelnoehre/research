from collections import deque
from fcapy.lattice import ConceptLattice


def _lectically_smaller(attributes: list, intent_a: set, intent_b: set) -> bool:
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
    
    for m in attributes:
        if m in intent_a and m in intent_b:
            continue
        if m not in intent_a and m not in intent_b:
            continue
        # m is the smallest differing element
        # A <L B iff A does NOT contain m
        return m not in intent_a
    
    return False

def compute_lectic_order(concepts: list, intents: dict, attributes: list) -> list:
    '''
    Sort all concepts by the lectic order on their intents without modifying the original list.
    '''
    # Create a shallow copy so we don't destroy the original 'concepts' order
    sorted_concepts = list(concepts)
    
    for i in range(1, len(sorted_concepts)):
        key = sorted_concepts[i]
        key_intent = intents[key]
        j = i - 1
        while j >= 0 and _lectically_smaller(attributes, key_intent, intents[sorted_concepts[j]]):
            sorted_concepts[j + 1] = sorted_concepts[j]
            j -= 1
        sorted_concepts[j + 1] = key

    return sorted_concepts

def all_intents(
        lattice: ConceptLattice
    ) -> dict:
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

def all_extents(
        lattice: ConceptLattice
    ) -> dict:
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