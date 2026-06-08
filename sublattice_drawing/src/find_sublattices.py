import networkx as nx
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


def all_maximal_chains(G):
    sources = [v for v in G if G.in_degree(v) == 0]
    sinks = [v for v in G if G.out_degree(v) == 0]
    chains = []
    for source in sources:
        for sink in sinks:
            chains.extend(nx.all_simple_paths(G, source, sink))
    return chains