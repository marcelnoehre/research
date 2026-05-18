from collections import deque

from data.parser import Parser
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
    Sort all concepts by the lectic order on their intents.

    Returns
    -------
    list
        Concept IDs sorted lectically
    '''
    
    for i in range(1, len(concepts)):
        key = concepts[i]
        key_intent = intents[key]
        j = i - 1
        while j >= 0 and _lectically_smaller(attributes, key_intent, intents[concepts[j]]):
            concepts[j + 1] = concepts[j]
            j -= 1
        concepts[j + 1] = key

    return concepts

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

type_mapping = {
    'hand_drawn': 'hand_drawn', 
    'sup_inf_attribute': 'sup_inf_attribute', 
    'sup_inf_double': 'sup_inf_doubly', 
    'dim_draw': 'dim_draw', 
    'dim_draw_double': 'dim_flux'
}
cxt_mapping = {
    'living_beings_and_water': 'living_beings_and_water', 
    'car': 'drive_concepts', 
    'Forum-Romanum': 'forum_romanum',
    'triangles': 'triangles' , 
    'convex-ordinal': 'convex_ordinal'
}
for i in range(1, 127):
    cxt_mapping[i] = str(i)

cxt_path = {
    'hand_drawn': 'm4', 
    'sup_inf_attribute': 'm4_original', 
    'sup_inf_double': 'm4', 
    'dim_draw': 'm4_dim_draw', 
    'dim_draw_double': 'm4_dim_draw'
}
parser = Parser()
for type_original in [
    'hand_drawn', 
    'sup_inf_attribute', 
    'sup_inf_double', 
    'dim_draw', 
    'dim_draw_double'
    ]:
    for cxt_original in [i for i in range(1, 127)] + [
        'living_beings_and_water', 
        'car', 
        'Forum-Romanum',
        'triangles', 
        'convex-ordinal'
    ]:
        _folder = cxt_path[type_original] if isinstance(cxt_original, (int, float)) else 'study_reduced'
        _cxt = parser.decode_cxt(f'cxt_original/{_folder}/{cxt_original}.cxt')
        _lat = ConceptLattice.from_context(_cxt)
        _concepts = list(_lat.to_networkx().nodes)
        _intents = all_intents(_lat)
        _attributes = _cxt.attribute_names
        _lectic_order = compute_lectic_order(_concepts, _intents, _attributes)
        with open(f'pos_original/{type_original}/{cxt_original}.pos', 'r') as f:
            positions = {
                c: tuple(map(float, line.split()[:2]))
                for c, line in enumerate(f) if line.strip()
            }

         
        pos = [f'{positions[c][0]} {positions[c][1]}' for c in _lectic_order]
        with open(f'pos_lectic/{type_mapping[type_original]}/{cxt_mapping[cxt_original]}.pos', 'w', encoding='utf-8') as f:
            f.write('\n'.join(pos))