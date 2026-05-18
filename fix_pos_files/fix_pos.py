from collections import deque

from data.parser import Parser
from fcapy.lattice import ConceptLattice

mirror = {
    'hand_drawn': [],
    'sup_inf_attribute': [3, 7, 10, 12, 15, 28, 32, 35, 36, 46, 48, 51, 55, 56, 57, 62, 68, 88, 92, 102, 106, 107, 'Forum-Romanum', 'living_beings_and_water', 'triangles'],
    'sup_inf_double': [7, 3, 10, 14, 16, 17, 18, 20, 25, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 61, 62, 64, 68, 71, 72, 80, 81, 82, 85, 86, 87, 88, 94, 96, 99, 100, 101, 102, 106, 107, 110, 112, 113, 114, 115, 117, 118, 119, 121, 123],
    'dim_draw': [7, 10, 12, 13, 15, 16, 29, 45, 46, 47, 53, 55, 56, 61, 62, 63, 64, 65, 70, 71, 72, 77, 79, 81, 86, 87, 88, 94, 96, 101, 104, 106, 116, 'car'],
    'dim_draw_double': [7, 10, 12, 13, 15, 16, 29, 45, 46, 47, 53, 55, 56, 61, 62, 63, 64, 65, 70, 71, 72, 77, 79, 81, 86, 87, 88, 94, 96, 101, 104, 106, 116, 'car']
}

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

def scaled_positions(cxt, type, size):
    with open(f'pos_original/{type}/{cxt}.pos', 'r') as f:
        coords = [tuple(map(float, line.split()[:2])) for line in f if line.strip()]

    if not coords:
        return []

    x_vals = [p[0] for p in coords]
    y_vals = [p[1] for p in coords]
    
    if isinstance(cxt, int) and cxt > 1:
        min_x, max_x = min(x_vals), max(x_vals)
        min_y_orig = min(y_vals)
        
        range_x = max_x - min_x
        scale = size / range_x if range_x != 0 else 1.0
        
        scaled_coords = [
            (-(size * 0.5) + (x - min_x) * scale, (y - min_y_orig) * scale)
            for x, y in coords
        ]
    else:
        scaled_coords = coords

    y_vals_scaled = [p[1] for p in scaled_coords]
    min_y_scaled, max_y_scaled = min(y_vals_scaled), max(y_vals_scaled)
    height = (max_y_scaled - min_y_scaled)

    if height > (size * 0.25):
        # Determine if mirroring is required for this context
        mirror_factor = -1.0 if cxt in mirror.get(type, []) else 1.0
        shrink_factor = (size * 0.25) / height
        
        processed_coords = [
            (x * shrink_factor * mirror_factor, (y - min_y_scaled) * shrink_factor)
            for x, y in scaled_coords
        ]
    else:
        # Simple normalization to Y=0
        processed_coords = [
            (x, (y - min_y_scaled))
            for x, y in scaled_coords
        ]

    final_min_y = min(p[1] for p in processed_coords)
    final_coords = {
        i: (p[0], p[1] - final_min_y)
        for i, p in enumerate(processed_coords)
    }

    return final_coords

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
        positions = scaled_positions(cxt_original, type_original, 10 if isinstance(cxt_original, (int, float)) else 20)
        concept_to_coord = {concept: positions[i] for i, concept in enumerate(_concepts)}
        pos = [f'{concept_to_coord[c][0]} {concept_to_coord[c][1]}' for c in _lectic_order]
        with open(f'pos_lectic/{type_mapping[type_original]}/{cxt_mapping[cxt_original]}.pos', 'w', encoding='utf-8') as f:
            f.write('\n'.join(pos))