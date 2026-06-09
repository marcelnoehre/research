def total_ordering(realizer) -> dict:
    return {
        node: (0, float(i))
        for i, node in enumerate(realizer[0])
    }

def S7_positions(sublattice, realizer):
    assert len(sublattice) == 1
    s7 = sublattice[0]
    positions = {}

    idx0 = {node: i for i, node in enumerate(realizer[0])}
    idx1 = {node: i for i, node in enumerate(realizer[1])}

    d, e = s7['d'], s7['e']
    factor = {}
    if idx0[d] > idx0[e] and idx1[d] < idx1[e]:
        factor = {
            'ad': -1.0,
            'ce': 1.0
        }
    else:
        factor = {
            'ad': 1.0,
            'ce': -1.0
        }

    positions[s7['bot']] = (0.0, 0.0)
    positions[s7['d']] = (factor['ad'], 1.0)
    positions[s7['e']] = (factor['ce'], 1.0)
    positions[s7['a']] = (factor['ad'], 2.0)
    positions[s7['c']] = (factor['ce'], 2.0)
    positions[s7['b']] = (0.0, 2.0)
    positions[s7['top']] = (0.0, 3.0)

    return positions