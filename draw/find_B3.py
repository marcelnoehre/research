from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx
from networkx.algorithms import isomorphism

from skip_doubly_irreducibles import skip_doubly_irreducibles

B3_ref = nx.DiGraph()
B3_ref.add_edges_from([
    ('bot', 'a'), ('bot', 'b'),  ('bot', 'c'),
    ('a',  'ab'), ('a',  'ac'),
    ('b',  'ab'), ('b',  'bc'),
    ('c',  'ac'), ('c',  'bc'),
    ('ab', 'top'),('ac', 'top'), ('bc', 'top'),
])

if __name__ == "__main__":
    file = 'Forum-Romanum'
    parser = Parser()
    cxt = parser.decode_cxt(f'../data/{file}.cxt')
    lat = ConceptLattice.from_context(cxt)
    G     = nx.transitive_reduction(lat.to_networkx())
    skip_doubly_irreducibles(G)

    seen = set()
    b3s  = []
    for iso in isomorphism.DiGraphMatcher(G, B3_ref).subgraph_monomorphisms_iter():
        key = frozenset(iso)
        if key in seen:
            continue
        seen.add(key)
        inv = {v: k for k, v in iso.items()}
        b3s.append(inv)

    print(f"Found {len(b3s)} B3 sublattices")
    for i, inv in enumerate(b3s):
        print(f"\nB3 #{i+1}")
        print(f"  bottom  : {inv['bot']}")
        print(f"  atoms   : {inv['a']}, {inv['b']}, {inv['c']}")
        print(f"  coatoms : {inv['ab']}, {inv['ac']}, {inv['bc']}")
        print(f"  top     : {inv['top']}")