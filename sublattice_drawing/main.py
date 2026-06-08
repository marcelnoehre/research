from data import Parser
from fcapy.lattice import ConceptLattice
from src.cut import *

file = '7'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

anchors, parts = find_anchor_pairs(G)

print(f"anchors (top -> bottom): {anchors}\n")
print(f"{len(parts)} part(s):")
for a, b, region in parts:
    inner = len(region) - 2
    print(f"A = {a}  ->  B = {b}   ({inner} node(s) enclosed)")
    print(f"    region: {sorted(region, key=str)}")