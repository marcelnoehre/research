import pandas as pd
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

if __name__ == "__main__":
    file = 'atlas'
    parser = Parser()
    ctx = parser.decode_cxt(f'{file}.cxt')
    
    down_arrows, up_arrows, double_arrows = standard_context(ctx)
    print('down_arrows:', down_arrows - double_arrows)
    print('up_arrows:', up_arrows - double_arrows)
    print('double_arrows', double_arrows)