import numpy as np
from src.utils.variables import Variables
from src.dim_flux.lgs import LinearEquationSolver

def _concept_pos(vars, concept, vectors) -> np.ndarray:
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
    base_vectors = np.array([vars.element_map[v] for v in (vars.extents[concept] | vars.intents[concept])])

    # (0, 0) if concept has no objects and no attributes
    if not base_vectors.size:
        return np.zeros(2)
    
    return np.sum(vectors[base_vectors], axis=0)

for name in ['Forum-Romanum', 'car', 'convex-ordinal', 'triangles', 'living_beings_and_water']:
    name = 'Forum-Romanum'
    cxt = f'evaluation/data/study_reduced/{name}.cxt'
    coordinates = {}
    with open(f'evaluation/positions/dim_flux/{str(name)}.pos', 'r') as f:
        for index, line in enumerate(f):
            parts = line.split()
            coordinates[index] = (float(parts[0]), float(parts[1]))

    vars = Variables(str(name), cxt, {
        'plot_si_graph': False,
        'si_graph_annotations':  False,
        'plot_initial_layout':  False,
        'initial_layout_annotations':  False,
        'plot_optimized_layout':  False,
        'optimized_layout_annotations':  False,
        'plot_individual_forces':  False,
        'plot_combined_forces':  False,
        'plot_gradients':  False,
        'plot_origin':  False
    })

    vars.coordinates = coordinates

    lgs = LinearEquationSolver(vars, coordinates)
    success, vector_vars = lgs.solve_linear_equations()
    if success:
        base_vectors = dict({})
        for v in vars.elements:
            if v in vars.G:
                base_vectors[v] = np.array([vector_vars[f'x_{v}'], vector_vars[f'y_{v}']])
            else:
                base_vectors[v] = np.array([-vector_vars[f'x_{v}'], -vector_vars[f'y_{v}']])

    base_vectors = {k: np.round(v) for k, v in base_vectors.items()}
    coordinates = {}
    for c in vars.concepts:
        coordinates[c] = _concept_pos(vars, c, np.array([base_vectors[v] for v in vars.elements]))

    pos = [f'{vars.coordinates[c][0]} {vars.coordinates[c][1]}' for c in vars.concepts]
    with open(f'evaluation/positions/dim_draw_double/{vars.name}.pos', 'w', encoding='utf-8') as f:
        f.write('\n'.join(pos))