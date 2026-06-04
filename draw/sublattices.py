from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx

file = 'Forum-Romanum'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_closure(lat.to_networkx())
