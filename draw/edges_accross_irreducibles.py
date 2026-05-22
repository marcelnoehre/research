from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx

file = '43'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

for node in G.nodes:
    preds = list(G.predecessors(node))   # nodes above
    succs = list(G.successors(node))     # nodes below
    is_doubly_irreducible = (len(preds) == 1 and len(succs) == 1)
    if is_doubly_irreducible:
        print(preds)
        print(succs)
