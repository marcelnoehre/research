from data import Parser
from fcapy.lattice import ConceptLattice
from src.realizer import SatRealizer
from src.cut import *
from src.refs import *
from src.find_sublattices import *

file = '3'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

anchors, parts = parts_as_lattices(G, lat)

print(f"anchors (top -> bottom): {anchors}\n")
print(f"{len(parts)} part(s):")
for a, b, region, sub_lat, index_map in parts:
    inner = len(region) - 2
    print(f"A = {a}  ->  B = {b}   ({inner} node(s) enclosed)")
    print(f"    region:    {sorted(region, key=str)}")
    print(f"    index_map: {index_map}")
    print(f"    sub_lat:   {len(sub_lat)} concepts")

    realizer = SatRealizer(sub_lat)
    print(realizer.realizer())

    print(find_sublattice(sub_lat.to_networkx(), N5_ref))