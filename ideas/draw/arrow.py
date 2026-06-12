import pandas as pd
import networkx as nx
from itertools import combinations
from data import Parser
from fcapy.lattice import ConceptLattice
from fcapy.context import FormalContext
from fcapy.context.converters import to_pandas


def standard_context(cxt):
    df = to_pandas(cxt)
    objects=list(df.index) 
    attributes=list(df.columns)
    intents = {
        g: set(df.columns[df.loc[g]]) for g in objects
    }
    extents={
        m: set(df.index[df[m]]) for m in attributes
    }

    down_arrows = set()
    up_arrows = set()
    double_arrows = set()
    for g in objects:
        for m in attributes:
            if df.loc[g,m]: 
                continue # already related

            # every object with strictly larger intent must contain m
            is_down = True
            for h in objects:
                if intents[g] < intents[h]:
                    if m not in intents[h]:
                        is_down = False
                        break

            # every attribute with strictly larger extent must contain g
            is_up = True
            for n in attributes:
                if extents[m] < extents[n]:
                    if g not in extents[n]:
                        is_up = False
                        break

            if is_down:
                down_arrows.add((g,m))
            if is_up:
                up_arrows.add((g,m))
            if is_down and is_up:
                double_arrows.add((g,m))

    return down_arrows, up_arrows, double_arrows


def atlas_decomposition(cxt):
    down_arrows, up_arrows, double_arrows = standard_context(cxt)

    df = to_pandas(cxt)
    objects = list(df.index)
    attributes = list(df.columns)

    # bipartite graph: objects and attributes linked by any arrow relation
    G = nx.Graph()
    G.add_nodes_from(('g', g) for g in objects)
    G.add_nodes_from(('m', m) for m in attributes)
    for g, m in down_arrows | up_arrows:
        G.add_edge(('g', g), ('m', m))

    blocks = []
    for comp in nx.connected_components(G):
        obj_group = frozenset(x[1] for x in comp if x[0] == 'g')
        attr_group = frozenset(x[1] for x in comp if x[0] == 'm')

        blk_double = frozenset((g, m) for g, m in double_arrows if g in obj_group)
        blk_down   = frozenset((g, m) for g, m in down_arrows   if g in obj_group) - blk_double
        blk_up     = frozenset((g, m) for g, m in up_arrows     if g in obj_group) - blk_double

        blocks.append({
            'objects':    obj_group,
            'attributes': attr_group,
            'down':       blk_down,
            'up':         blk_up,
            'double':     blk_double,
        })

    # stable ordering: larger blocks first, then lexicographic on sorted objects
    blocks.sort(key=lambda b: (-len(b['objects']) - len(b['attributes']),
                               sorted(b['objects']), sorted(b['attributes'])))
    return blocks


if __name__ == "__main__":
    file = 'atlas'
    parser = Parser()
    ctx = parser.decode_cxt(f'{file}.cxt')
    
    down_arrows, up_arrows, double_arrows = standard_context(ctx)
    print('down_arrows:', down_arrows - double_arrows)
    print('up_arrows:', up_arrows - double_arrows)
    print('double_arrows', double_arrows)

    print()
    blocks = atlas_decomposition(ctx)
    print(f'atlas decomposition: {len(blocks)} block(s)')
    for i, b in enumerate(blocks):
        print(f'\n  block {i}')
        print(f'    objects:    {sorted(b["objects"])}')
        print(f'    attributes: {sorted(b["attributes"])}')
        if b['double']: print(f'    double:     {sorted(b["double"])}')
        if b['down']:   print(f'    down only:  {sorted(b["down"])}')
        if b['up']:     print(f'    up only:    {sorted(b["up"])}')