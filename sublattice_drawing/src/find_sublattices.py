from networkx.algorithms import isomorphism

def find_sublattice(G, ref):
    seen = set()
    sub  = []
    for iso in isomorphism.DiGraphMatcher(G, ref).subgraph_monomorphisms_iter():
        key = frozenset(iso)
        if key in seen:
            continue
        seen.add(key)
        inv = {v: k for k, v in iso.items()}
        sub.append(inv)

    return sub