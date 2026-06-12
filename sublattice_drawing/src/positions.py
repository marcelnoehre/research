import numpy as np
from src.refs import CANONICAL_ANGLES

def total_ordering(realizer) -> dict:
    return {
        node: (0, float(i))
        for i, node in enumerate(realizer[0])
    }


def _angle_magnitudes(n_mag):
    usable = sorted(a for a in CANONICAL_ANGLES if 0 < a < 90)
    
    if n_mag == 0:
        return []
    
    if n_mag <= len(usable):
        mid = len(usable) // 2
        lo  = mid - (n_mag - 1) // 2
        return usable[lo:lo + n_mag]
    
    return [(k + 1) / (n_mag + 1) * 90.0 for k in range(n_mag)]

def simple_chain_positions(intervals):
    chains = sorted(list(intervals.values())[0], key=len, reverse=True) # bottom-up chains
    n_chains = len(chains)
    H = max(len(c) for c in chains) - 1 # longest chain

    has_center = (n_chains % 2 == 1)
    rest = chains[1:] if has_center else chains # rest always even
    mags = _angle_magnitudes(len(rest) // 2)

    lanes = [(0, 0.0)] if has_center else []
    for i in range(len(rest)):
        side = -1 if i % 2 == 0 else +1 # longer chains left, then alternate
        lanes.append((side, mags[i // 2]))

    positions = {}
    for chain, (side, angle) in zip(chains, lanes):
        n = len(chain)
        if n <= 2:
            for j, e in enumerate(chain):
                positions[e] = (0.0, j * H / (n - 1))
            continue

        h = H / (n - 1)
        X = side * h * np.tan(np.radians(angle))
        for j, e in enumerate(chain):
            y = j * H / (n - 1)
            x = 0.0 if (j == 0 or j == n - 1) else X
            positions[e] = (x, y)

    return positions

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