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


def parallel_intervals(G):
    """
    Find all (start, end) pairs connected by more than one path.
    Returns a dict mapping (start, end) -> list of paths, sorted by
    number of paths descending.
    """
    chains = all_maximal_chains(G)
    groups = {}
    for chain in chains:
        for i in range(len(chain)):
            for j in range(i + 2, len(chain) + 1):
                sub = tuple(chain[i:j])
                key = (sub[0], sub[-1])
                groups.setdefault(key, set()).add(sub)
    groups = {k: [list(p) for p in v] for k, v in groups.items() if len(v) > 1}
    return dict(sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True))