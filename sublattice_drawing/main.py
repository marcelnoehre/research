from data import Parser
from fcapy.lattice import ConceptLattice
from src.realizer import SatRealizer
from src.cut import *
from src.refs import *
from src.find_sublattices import *
from src.positions import *

file = '3'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

anchors, parts = parts_as_lattices(G, lat)

positions = {}

print(f"anchors (top -> bottom): {anchors}\n")
print(f"{len(parts)} part(s):")
for a, b, region, sub_lat, index_map in parts:
    inner = len(region) - 2
    print(f"A = {a}  ->  B = {b}   ({inner} node(s) enclosed)")
    print(f"    region:    {sorted(region, key=str)}")
    print(f"    index_map: {index_map}")
    print(f"    sub_lat:   {len(sub_lat)} concepts")

    sat_realizer = SatRealizer(sub_lat)
    
    dim, realizer = sat_realizer.realizer()

    if dim == 1:
        chain_positions = total_ordering(realizer)
        # todo append chain positions based on the lower cut element if not bot
    elif dim == 2:
        find_sublattice(sub_lat.to_networkx(), N5_ref)
    elif dim == 3:
        find_sublattice(sub_lat.to_networkx(), B3_ref)
    else:
        raise ValueError('Not implemented')